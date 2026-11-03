// KDA RecoverSSM verify and state commit for an FP32 SSM state with head_dim
// 128 on SM10x, with the contract of the Triton kernels in
// vllm/models/kimi_k3/nvidia/ops/recoverssm.py.
//
// One CTA owns one 128x128 FP32 state (64 KiB). Each of its 256 threads keeps a
// 4-row x 16-column tile in registers: lane bits [4:3] pick 4 consecutive rows
// within the warp's 16 rows, and lane bits [2:0] pick the columns
// 32 * m + 4 * g + [0, 4), m < 4. The 8 lanes sharing rows read 128 contiguous
// bytes of a row per load, and their per-column operands in shared memory are 8
// consecutive float4, so neither access conflicts. A row sum is a
// reduce-scatter over those 8 lanes. The state math uses the packed FP32x2 FMA.

#include <cstdint>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../torch_utils.h"

namespace {

constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kMaxQ = 8;  // verified tokens per step; one staging warp each
constexpr int kRows = 4;
constexpr int kChunks = 4;

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.f / (1.f + __expf(-x));
}

__device__ __forceinline__ float2 lo(float4 x) { return make_float2(x.x, x.y); }
__device__ __forceinline__ float2 hi(float4 x) { return make_float2(x.z, x.w); }

// exp of the KDA gate of one key column.
__device__ __forceinline__ float kda_decay(float raw_g, float dt_bias, float A,
                                           float lower_bound,
                                           bool use_lower_bound) {
  float const x = raw_g + dt_bias;
  if (use_lower_bound) return __expf(lower_bound * sigmoidf(A * x));
  float const softplus = x > 20.f ? x : __logf(1.f + __expf(x));
  return __expf(-A * softplus);
}

// One warp per token, 4 key columns per lane: the token's L2-normalized key,
// its decay and, when q_row is given, its normalized query scaled by K^-0.5.
__device__ __forceinline__ void stage_token(__nv_bfloat16 const* k_row,
                                            __nv_bfloat16 const* g_row,
                                            __nv_bfloat16 const* q_row,
                                            float const* dt_bias, float A,
                                            float lower_bound,
                                            bool use_lower_bound, int lane,
                                            float* nk, float* qn, float* dec) {
  int const c = lane * 4;
  float kv[4], gv[4], qv[4] = {0.f, 0.f, 0.f, 0.f};
  float kk = 0.f, qq = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    kv[i] = bf2f(k_row[c + i]);
    gv[i] = bf2f(g_row[c + i]);
    kk += kv[i] * kv[i];
    if (q_row != nullptr) {
      qv[i] = bf2f(q_row[c + i]);
      qq += qv[i] * qv[i];
    }
  }
  float const k_scale = rsqrtf(warp_sum(kk) + 1e-6f);
  float const q_scale = rsqrtf(warp_sum(qq) + 1e-6f) * rsqrtf(float(kDim));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    nk[c + i] = kv[i] * k_scale;
    if (q_row != nullptr) qn[c + i] = qv[i] * q_scale;
    dec[c + i] =
        kda_decay(gv[i], dt_bias[c + i], A, lower_bound, use_lower_bound);
  }
}

// Sums p[i] over the 8 lanes sharing rows; lane g gets the total of row g >> 1.
__device__ __forceinline__ float reduce_rows(float const (&p)[kRows], int g) {
  bool const h = g & 4, m = g & 2;
  float const a =
      (h ? p[2] : p[0]) + __shfl_xor_sync(0xffffffffu, h ? p[0] : p[2], 4);
  float const b =
      (h ? p[3] : p[1]) + __shfl_xor_sync(0xffffffffu, h ? p[1] : p[3], 4);
  float const r = (m ? b : a) + __shfl_xor_sync(0xffffffffu, m ? a : b, 2);
  return r + __shfl_xor_sync(0xffffffffu, r, 1);
}

// s[i][j] holds row r0 + i, columns 32 * (j / 2) + 4 * g + 2 * (j % 2) + [0,
// 2).
struct Tile {
  float2 s[kRows][2 * kChunks];

  __device__ __forceinline__ void load(float const* rows, int g) {
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int m = 0; m < kChunks; ++m) {
        float4 const x = __ldcs(
            reinterpret_cast<float4 const*>(rows + i * kDim + 32 * m + 4 * g));
        s[i][2 * m] = lo(x);
        s[i][2 * m + 1] = hi(x);
      }
  }

  __device__ __forceinline__ void store(float* rows, int g) const {
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int m = 0; m < kChunks; ++m) {
        float2 const a = s[i][2 * m], b = s[i][2 * m + 1];
        __stcs(reinterpret_cast<float4*>(rows + i * kDim + 32 * m + 4 * g),
               make_float4(a.x, a.y, b.x, b.y));
      }
  }
};

struct VerifyStrides {
  int64_t q, k, v, g, beta, out;
  int64_t state_block, state_head;
  int64_t corr_row, corr_head, corr_pos;
  int64_t kg_row, kg_head, kg_pos;
  int64_t idx;
};

// One CTA per (request, head). The checkpoint is read, never written; the
// corrections u_t and the raw keys and gates are recorded for the commit.
__global__ void __launch_bounds__(kThreads, 2) kda_recoverssm_verify_kernel(
    __nv_bfloat16 const* __restrict__ q, __nv_bfloat16 const* __restrict__ k,
    __nv_bfloat16 const* __restrict__ v,
    __nv_bfloat16 const* __restrict__ raw_g,
    __nv_bfloat16 const* __restrict__ raw_beta, float const* __restrict__ A_log,
    float const* __restrict__ dt_bias, float const* __restrict__ state,
    float* __restrict__ correction, __nv_bfloat16* __restrict__ key_gate,
    __nv_bfloat16* __restrict__ out, int const* __restrict__ query_start_loc,
    int const* __restrict__ state_indices, float lower_bound,
    bool use_lower_bound, int null_block_id, int spec_query_len,
    VerifyStrides st) {
  int const b = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  int const warp = tid >> 5, lane = tid & 31, g = lane & 7;
  int const r0 = warp * 16 + (lane >> 3) * kRows;
  int64_t const bos = query_start_loc[b];
  int const qlen = min(int(query_start_loc[b + 1] - bos), spec_query_len);
  int const sidx = state_indices[b * st.idx];

  if (sidx <= null_block_id) {
    for (int t = 0; t < qlen; ++t)
      if (tid < kDim)
        out[(bos + t) * st.out + h * kDim + tid] = __float2bfloat16(0.f);
    return;
  }

  __shared__ __align__(16) float s_nk[kMaxQ][kDim];
  __shared__ __align__(16) float s_qn[kMaxQ][kDim];
  __shared__ __align__(16) float s_dec[kMaxQ][kDim];
  __shared__ float s_v[kMaxQ][kDim], s_beta[kMaxQ];

  // The state loads are in flight while the tokens are staged.
  Tile tile;
  tile.load(state + sidx * st.state_block + h * st.state_head + r0 * kDim, g);

  if (warp < qlen) {
    int const t = warp;
    int64_t const tok = bos + t;
    __nv_bfloat16 const* k_row = k + tok * st.k + h * kDim;
    __nv_bfloat16 const* g_row = raw_g + tok * st.g + h * kDim;
    stage_token(k_row, g_row, q + tok * st.q + h * kDim, dt_bias + h * kDim,
                __expf(A_log[h]), lower_bound, use_lower_bound, lane, s_nk[t],
                s_qn[t], s_dec[t]);
    __nv_bfloat16* kg =
        key_gate + b * st.kg_row + h * st.kg_head + t * st.kg_pos;
    int const c = lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kg[c + i] = k_row[c + i];
      kg[kDim + c + i] = g_row[c + i];
      s_v[t][c + i] = bf2f(v[tok * st.v + h * kDim + c + i]);
    }
    if (lane == 0) s_beta[t] = sigmoidf(bf2f(raw_beta[tok * st.beta + h]));
  }
  __syncthreads();

  int const ri = g >> 1;
  for (int t = 0; t < qlen; ++t) {
    float2 nk[2 * kChunks], dec[2 * kChunks], qn[2 * kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
      float4 const x =
          *reinterpret_cast<float4 const*>(&s_nk[t][32 * m + 4 * g]);
      float4 const d =
          *reinterpret_cast<float4 const*>(&s_dec[t][32 * m + 4 * g]);
      nk[2 * m] = lo(x);
      nk[2 * m + 1] = hi(x);
      dec[2 * m] = lo(d);
      dec[2 * m + 1] = hi(d);
    }
    // u_t = (v_t - (S d_t) k_t) beta_t
    float p[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < 2 * kChunks; ++j) {
        tile.s[i][j] = __fmul2_rn(tile.s[i][j], dec[j]);
        acc = __ffma2_rn(tile.s[i][j], nk[j], acc);
      }
      p[i] = acc.x + acc.y;
    }
    float const corr = (s_v[t][r0 + ri] - reduce_rows(p, g)) * s_beta[t];

    // S += u_t k_t^T, o_t = S q_t
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
      float4 const x =
          *reinterpret_cast<float4 const*>(&s_qn[t][32 * m + 4 * g]);
      qn[2 * m] = lo(x);
      qn[2 * m + 1] = hi(x);
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      float const u = __shfl_sync(0xffffffffu, corr, (lane & ~7) | (2 * i));
      float2 const u2 = make_float2(u, u);
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < 2 * kChunks; ++j) {
        tile.s[i][j] = __ffma2_rn(u2, nk[j], tile.s[i][j]);
        acc = __ffma2_rn(tile.s[i][j], qn[j], acc);
      }
      p[i] = acc.x + acc.y;
    }
    float const o = reduce_rows(p, g);
    if ((g & 1) == 0) {
      out[(bos + t) * st.out + h * kDim + r0 + ri] = __float2bfloat16(o);
      correction[b * st.corr_row + h * st.corr_head + t * st.corr_pos + r0 +
                 ri] = corr;
    }
  }
}

struct CommitStrides {
  int64_t state_head, corr_head, corr_pos, kg_head, kg_pos;
  int64_t a_layer, a_head, dt_layer, dt_head, idx;
};

// One CTA per (request, layer, head) folds the accepted tokens into the state
// in place: S <- S prod_t d_t + sum_t (u_t k_t^T) prod_{s>t} d_s. When prefix
// caching keeps the state at the block boundary, the tokens before it are
// folded and stored first, then the rest.
__global__ void __launch_bounds__(kThreads, 2) kda_recoverssm_commit_kernel(
    int64_t const* __restrict__ state_base,
    int64_t const* __restrict__ state_block_stride,
    int64_t const* __restrict__ corr_base,
    int64_t const* __restrict__ corr_row_stride,
    int64_t const* __restrict__ kg_base,
    int64_t const* __restrict__ kg_row_stride, float const* __restrict__ A_log,
    float const* __restrict__ dt_bias, int const* __restrict__ state_indices,
    int const* __restrict__ commit_lens,
    int const* __restrict__ final_state_indices,
    int const* __restrict__ boundary_state_indices,
    int const* __restrict__ boundary_recovery_lens, float lower_bound,
    bool use_lower_bound, int null_block_id, int num_heads, bool align,
    CommitStrides st) {
  int const b = blockIdx.x, l = blockIdx.y / num_heads,
            h = blockIdx.y % num_heads;
  int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane & 7;
  int const r0 = warp * 16 + (lane >> 3) * kRows;

  int const src_idx = state_indices[b * st.idx];
  int const clen = commit_lens[b];
  int const fin_idx = final_state_indices[b];
  if (src_idx <= null_block_id || clen == 0 || fin_idx <= null_block_id) return;
  int const bnd_idx = align ? boundary_state_indices[b] : null_block_id;
  bool const write_boundary = bnd_idx > null_block_id;
  int const blen = write_boundary ? min(boundary_recovery_lens[b], clen) : 0;

  float* state =
      reinterpret_cast<float*>(state_base[l]) + h * st.state_head + r0 * kDim;
  int64_t const block_stride = state_block_stride[l];
  float const* corr = reinterpret_cast<float const*>(corr_base[l]) +
                      b * corr_row_stride[l] + h * st.corr_head;
  __nv_bfloat16 const* kg = reinterpret_cast<__nv_bfloat16 const*>(kg_base[l]) +
                            b * kg_row_stride[l] + h * st.kg_head;

  // Within each segment [0, blen) and [blen, clen): w_t = k_t prod_{s>t} d_s,
  // and the segment's decay prod_t d_t.
  __shared__ __align__(16) float s_w[kMaxQ][kDim];
  __shared__ __align__(16) float s_dec[kMaxQ][kDim];
  __shared__ __align__(16) float s_dseg[2][kDim];
  __shared__ __align__(16) float s_u[kMaxQ][kDim];

  Tile tile;
  tile.load(state + int64_t(src_idx) * block_stride, g);

  if (warp < clen) {
    int const t = warp;
    stage_token(kg + t * st.kg_pos, kg + t * st.kg_pos + kDim, nullptr,
                dt_bias + l * st.dt_layer + h * st.dt_head,
                __expf(A_log[l * st.a_layer + h * st.a_head]), lower_bound,
                use_lower_bound, lane, s_w[t], nullptr, s_dec[t]);
    *reinterpret_cast<float4*>(&s_u[t][lane * 4]) =
        *reinterpret_cast<float4 const*>(corr + t * st.corr_pos + lane * 4);
  }
  __syncthreads();
  if (tid < kDim) {
    float after = 1.f;
    for (int t = clen - 1; t >= blen; --t) {
      s_w[t][tid] *= after;
      after *= s_dec[t][tid];
    }
    s_dseg[1][tid] = after;
    after = 1.f;
    for (int t = blen - 1; t >= 0; --t) {
      s_w[t][tid] *= after;
      after *= s_dec[t][tid];
    }
    s_dseg[0][tid] = after;
  }
  __syncthreads();

  auto fold = [&](int t0, int t1, float const* d, int dst_idx) {
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
      float4 const x = *reinterpret_cast<float4 const*>(&d[32 * m + 4 * g]);
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        tile.s[i][2 * m] = __fmul2_rn(tile.s[i][2 * m], lo(x));
        tile.s[i][2 * m + 1] = __fmul2_rn(tile.s[i][2 * m + 1], hi(x));
      }
    }
    for (int t = t0; t < t1; ++t) {
      float4 const u = *reinterpret_cast<float4 const*>(&s_u[t][r0]);
      float const uu[kRows] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int m = 0; m < kChunks; ++m) {
        float4 const x =
            *reinterpret_cast<float4 const*>(&s_w[t][32 * m + 4 * g]);
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          float2 const u2 = make_float2(uu[i], uu[i]);
          tile.s[i][2 * m] = __ffma2_rn(u2, lo(x), tile.s[i][2 * m]);
          tile.s[i][2 * m + 1] = __ffma2_rn(u2, hi(x), tile.s[i][2 * m + 1]);
        }
      }
    }
    tile.store(state + int64_t(dst_idx) * block_stride, g);
  };
  if (write_boundary) fold(0, blen, s_dseg[0], bnd_idx);
  fold(blen, clen, s_dseg[1], fin_idx);
}

}  // namespace

void kda_recoverssm_verify(
    torch::stable::Tensor const& q, torch::stable::Tensor const& k,
    torch::stable::Tensor const& v, torch::stable::Tensor const& raw_g,
    torch::stable::Tensor const& raw_beta, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias, torch::stable::Tensor const& state,
    torch::stable::Tensor& correction, torch::stable::Tensor& key_gate,
    torch::stable::Tensor& out, torch::stable::Tensor const& query_start_loc,
    torch::stable::Tensor const& state_indices,
    std::optional<double> lower_bound, int64_t null_block_id,
    int64_t spec_query_len) {
  using torch::headeronly::ScalarType;
  // q, k, v, raw_g, out: [1, T, H, 128]; raw_beta: [1, T, H];
  // state: [blocks, H, 128, 128]; correction: [reqs, H, Q, 128];
  // key_gate: [reqs, H, Q, 256].
  auto const bf16 = [](torch::stable::Tensor const& t) {
    return t.scalar_type() == ScalarType::BFloat16;
  };
  STD_TORCH_CHECK(bf16(q) && bf16(k) && bf16(v) && bf16(raw_g) &&
                      bf16(raw_beta) && bf16(key_gate) && bf16(out),
                  "KDA RecoverSSM CUDA verify needs bfloat16 activations");
  STD_TORCH_CHECK(state.scalar_type() == ScalarType::Float &&
                      correction.scalar_type() == ScalarType::Float &&
                      a_log.scalar_type() == ScalarType::Float &&
                      dt_bias.scalar_type() == ScalarType::Float,
                  "KDA RecoverSSM CUDA verify needs a float32 state");
  STD_TORCH_CHECK(q.size(3) == kDim && v.size(3) == kDim,
                  "KDA RecoverSSM CUDA verify needs head_dim 128");
  STD_TORCH_CHECK(spec_query_len <= kMaxQ,
                  "KDA RecoverSSM CUDA verify takes at most 8 tokens per step");
  STD_TORCH_CHECK(
      state.stride(3) == 1 && state.stride(2) == kDim &&
          state.stride(1) % 4 == 0 && state.stride(0) % 4 == 0,
      "KDA RecoverSSM CUDA verify needs 16-byte aligned state rows");
  STD_TORCH_CHECK(correction.stride(3) == 1 && key_gate.stride(3) == 1,
                  "KDA RecoverSSM CUDA verify needs contiguous records");
  STD_TORCH_CHECK(query_start_loc.scalar_type() == ScalarType::Int &&
                      state_indices.scalar_type() == ScalarType::Int,
                  "KDA RecoverSSM CUDA verify needs int32 metadata");
  int64_t const batch = state_indices.size(0);
  if (batch == 0) return;

  VerifyStrides const st{
      q.stride(1),          k.stride(1),          v.stride(1),
      raw_g.stride(1),      raw_beta.stride(1),   out.stride(1),
      state.stride(0),      state.stride(1),      correction.stride(0),
      correction.stride(1), correction.stride(2), key_gate.stride(0),
      key_gate.stride(1),   key_gate.stride(2),   state_indices.stride(0)};
  torch::stable::accelerator::DeviceGuard const device_guard(
      q.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream(q.get_device_index());
  dim3 const grid(batch, q.size(2));
  kda_recoverssm_verify_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16 const*>(q.data_ptr()),
      static_cast<__nv_bfloat16 const*>(k.data_ptr()),
      static_cast<__nv_bfloat16 const*>(v.data_ptr()),
      static_cast<__nv_bfloat16 const*>(raw_g.data_ptr()),
      static_cast<__nv_bfloat16 const*>(raw_beta.data_ptr()),
      static_cast<float const*>(a_log.data_ptr()),
      static_cast<float const*>(dt_bias.data_ptr()),
      static_cast<float const*>(state.data_ptr()),
      static_cast<float*>(correction.data_ptr()),
      static_cast<__nv_bfloat16*>(key_gate.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int const*>(query_start_loc.data_ptr()),
      static_cast<int const*>(state_indices.data_ptr()),
      static_cast<float>(lower_bound.value_or(0.0)), lower_bound.has_value(),
      static_cast<int>(null_block_id), static_cast<int>(spec_query_len), st);
  cudaError_t const error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, "KDA RecoverSSM verify launch failed: ",
                  cudaGetErrorString(error));
}

void kda_recoverssm_commit(torch::stable::Tensor& state_ref,
                           torch::stable::Tensor const& state_base_addrs,
                           torch::stable::Tensor const& state_block_strides,
                           torch::stable::Tensor const& correction_ref,
                           torch::stable::Tensor const& correction_base_addrs,
                           torch::stable::Tensor const& correction_row_strides,
                           torch::stable::Tensor const& key_gate_ref,
                           torch::stable::Tensor const& key_gate_base_addrs,
                           torch::stable::Tensor const& key_gate_row_strides,
                           torch::stable::Tensor const& a_log,
                           torch::stable::Tensor const& dt_bias,
                           torch::stable::Tensor const& state_indices,
                           torch::stable::Tensor const& commit_lens,
                           torch::stable::Tensor const& final_state_indices,
                           torch::stable::Tensor const& boundary_state_indices,
                           torch::stable::Tensor const& boundary_recovery_lens,
                           std::optional<double> lower_bound,
                           int64_t null_block_id, int64_t batch, bool align) {
  using torch::headeronly::ScalarType;
  // *_ref: layer 0's tensor, whose strides every layer shares.
  STD_TORCH_CHECK(state_ref.scalar_type() == ScalarType::Float &&
                      state_ref.size(2) == kDim && state_ref.size(3) == kDim &&
                      state_ref.stride(3) == 1 && state_ref.stride(2) == kDim &&
                      state_ref.stride(1) % 4 == 0,
                  "KDA RecoverSSM CUDA commit needs a float32 state with "
                  "head_dim 128 and 16-byte aligned rows");
  STD_TORCH_CHECK(
      key_gate_ref.scalar_type() == ScalarType::BFloat16 &&
          key_gate_ref.stride(3) == 1 && correction_ref.stride(3) == 1 &&
          correction_ref.stride(2) % 4 == 0 && dt_bias.stride(2) == 1,
      "KDA RecoverSSM CUDA commit needs contiguous records");
  STD_TORCH_CHECK(state_indices.scalar_type() == ScalarType::Int,
                  "KDA RecoverSSM CUDA commit needs int32 state indices");
  if (batch == 0) return;

  int64_t const layers = state_base_addrs.size(0), heads = state_ref.size(1);
  CommitStrides const st{state_ref.stride(1),      correction_ref.stride(1),
                         correction_ref.stride(2), key_gate_ref.stride(1),
                         key_gate_ref.stride(2),   a_log.stride(0),
                         a_log.stride(1),          dt_bias.stride(0),
                         dt_bias.stride(1),        state_indices.stride(0)};
  torch::stable::accelerator::DeviceGuard const device_guard(
      state_ref.get_device_index());
  cudaStream_t const stream =
      get_current_cuda_stream(state_ref.get_device_index());
  dim3 const grid(batch, layers * heads);
  kda_recoverssm_commit_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<int64_t const*>(state_base_addrs.data_ptr()),
      static_cast<int64_t const*>(state_block_strides.data_ptr()),
      static_cast<int64_t const*>(correction_base_addrs.data_ptr()),
      static_cast<int64_t const*>(correction_row_strides.data_ptr()),
      static_cast<int64_t const*>(key_gate_base_addrs.data_ptr()),
      static_cast<int64_t const*>(key_gate_row_strides.data_ptr()),
      static_cast<float const*>(a_log.data_ptr()),
      static_cast<float const*>(dt_bias.data_ptr()),
      static_cast<int const*>(state_indices.data_ptr()),
      static_cast<int const*>(commit_lens.data_ptr()),
      static_cast<int const*>(final_state_indices.data_ptr()),
      static_cast<int const*>(boundary_state_indices.data_ptr()),
      static_cast<int const*>(boundary_recovery_lens.data_ptr()),
      static_cast<float>(lower_bound.value_or(0.0)), lower_bound.has_value(),
      static_cast<int>(null_block_id), static_cast<int>(heads), align, st);
  cudaError_t const error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, "KDA RecoverSSM commit launch failed: ",
                  cudaGetErrorString(error));
}
