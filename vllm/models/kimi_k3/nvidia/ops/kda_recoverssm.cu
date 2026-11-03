// KDA RecoverSSM verify and commit for an FP32 SSM state with head_dim 128 (sm_100).
//
// One CTA owns one 128x128 FP32 state (64 KiB). Each of its 256 threads keeps a 4-row x 16-column
// tile in registers: lane bits [4:3] pick 4 consecutive rows within the warp's 16 rows, and lane bits
// [2:0] pick the columns 32*m + 4*g + [0, 4), m < 4. The 8 lanes that share rows read 128 contiguous
// bytes of each row per load, and their per-column operands from shared memory are 8 consecutive
// float4, so neither access conflicts. A row sum is a reduce-scatter over those 8 lanes.
//
// The per-token vectors (normalized q and k, decay, beta) do not depend on the state; one warp per
// token stages them in shared memory while the state loads are in flight. The state math uses the
// packed FP32x2 FMA of sm_100.

#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kDim = 128;      // key_dim == value_dim
constexpr int kThreads = 256;
constexpr int kMaxQ = 8;       // speculative window (1 + num_speculative_tokens), one warp each
constexpr int kRows = 4;       // rows per thread
constexpr int kChunks = 4;     // float4 column chunks per thread

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float2 lo(float4 x) { return make_float2(x.x, x.y); }
__device__ __forceinline__ float2 hi(float4 x) { return make_float2(x.z, x.w); }

// exp of the KDA gate of one key column.
__device__ __forceinline__ float kda_decay(float raw_g, float dt_bias, float A, float lower_bound,
                                           bool use_lower_bound) {
  const float x = raw_g + dt_bias;
  if (use_lower_bound) return __expf(lower_bound * sigmoidf(A * x));
  const float softplus = x > 20.f ? x : __logf(1.f + __expf(x));
  return __expf(-A * softplus);
}

// One warp per token, 4 key columns per lane: the token's normalized key, decay and, when q_row is
// given, its normalized query scaled by K^-0.5.
__device__ __forceinline__ void stage_token(const __nv_bfloat16* k_row, const __nv_bfloat16* g_row,
                                            const __nv_bfloat16* q_row, const float* dt_bias,
                                            float A, float lower_bound, bool use_lower_bound,
                                            int lane, float* nk, float* qn, float* dec) {
  const int c = lane * 4;
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
  const float k_scale = rsqrtf(warp_sum(kk) + 1e-6f);
  const float q_scale = rsqrtf(warp_sum(qq) + 1e-6f) * rsqrtf(float(kDim));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    nk[c + i] = kv[i] * k_scale;
    if (q_row != nullptr) qn[c + i] = qv[i] * q_scale;
    dec[c + i] = kda_decay(gv[i], dt_bias[c + i], A, lower_bound, use_lower_bound);
  }
}

// Sums p[i] over the 8 lanes sharing rows; lane g gets the total of row g >> 1.
__device__ __forceinline__ float reduce_rows(const float (&p)[kRows], int g) {
  const bool h = g & 4, m = g & 2;
  const float a = (h ? p[2] : p[0]) + __shfl_xor_sync(0xffffffffu, h ? p[0] : p[2], 4);
  const float b = (h ? p[3] : p[1]) + __shfl_xor_sync(0xffffffffu, h ? p[1] : p[3], 4);
  float r = (m ? b : a) + __shfl_xor_sync(0xffffffffu, m ? a : b, 2);
  return r + __shfl_xor_sync(0xffffffffu, r, 1);
}

// The thread's tile: s[i][j] holds row r0 + i, columns 32 * (j / 2) + 4 * g + 2 * (j % 2) + [0, 2).
struct Tile {
  float2 s[kRows][2 * kChunks];

  __device__ __forceinline__ void load(const float* rows, int g) {
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int m = 0; m < kChunks; ++m) {
        const float4 x = __ldcs(reinterpret_cast<const float4*>(rows + i * kDim + 32 * m + 4 * g));
        s[i][2 * m] = lo(x);
        s[i][2 * m + 1] = hi(x);
      }
  }
};

template <typename IdxT>
__global__ void __launch_bounds__(kThreads, 2) verify_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ raw_g,
    const __nv_bfloat16* __restrict__ raw_beta, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, const float* __restrict__ state,
    float* __restrict__ correction, __nv_bfloat16* __restrict__ key_gate,
    __nv_bfloat16* __restrict__ out, const int* __restrict__ query_start_loc,
    const IdxT* __restrict__ state_indices, float lower_bound, bool use_lower_bound,
    int64_t null_block_id, int64_t s_q, int64_t s_k, int64_t s_v, int64_t s_g, int64_t s_beta,
    int64_t s_state_block, int64_t s_state_head, int64_t s_corr_row, int64_t s_corr_head,
    int64_t s_corr_pos, int64_t s_kg_row, int64_t s_kg_head, int64_t s_kg_pos, int64_t s_out,
    int64_t s_idx, int spec_query_len) {
  const int b = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane & 7;
  const int r0 = warp * 16 + (lane >> 3) * kRows;
  const int64_t bos = query_start_loc[b];
  const int qlen = min(int(query_start_loc[b + 1] - bos), spec_query_len);
  const int64_t sidx = int64_t(state_indices[b * s_idx]);

  if (sidx <= null_block_id) {
    for (int t = 0; t < qlen; ++t)
      if (tid < kDim) out[(bos + t) * s_out + h * kDim + tid] = __float2bfloat16(0.f);
    return;
  }

  __shared__ __align__(16) float s_nk[kMaxQ][kDim];
  __shared__ __align__(16) float s_qn[kMaxQ][kDim];
  __shared__ __align__(16) float s_dec[kMaxQ][kDim];
  __shared__ float s_val[kMaxQ][kDim], s_gain[kMaxQ];

  Tile tile;
  tile.load(state + sidx * s_state_block + h * s_state_head + r0 * kDim, g);

  const float A = __expf(A_log[h]);
  if (warp < qlen) {
    const int t = warp;
    const int64_t tok = bos + t;
    const __nv_bfloat16* k_row = k + tok * s_k + h * kDim;
    const __nv_bfloat16* g_row = raw_g + tok * s_g + h * kDim;
    stage_token(k_row, g_row, q + tok * s_q + h * kDim, dt_bias + h * kDim, A, lower_bound,
                use_lower_bound, lane, s_nk[t], s_qn[t], s_dec[t]);
    // The commit after sampling recomputes the decay and the key normalization from these.
    __nv_bfloat16* kg = key_gate + b * s_kg_row + h * s_kg_head + t * s_kg_pos;
    const int c = lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kg[c + i] = k_row[c + i];
      kg[kDim + c + i] = g_row[c + i];
      s_val[t][c + i] = bf2f(v[tok * s_v + h * kDim + c + i]);
    }
    if (lane == 0) s_gain[t] = sigmoidf(bf2f(raw_beta[tok * s_beta + h]));
  }
  __syncthreads();

  for (int t = 0; t < qlen; ++t) {
    float2 nk[2 * kChunks], dec[2 * kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
      const float4 x = *reinterpret_cast<const float4*>(&s_nk[t][32 * m + 4 * g]);
      const float4 d = *reinterpret_cast<const float4*>(&s_dec[t][32 * m + 4 * g]);
      nk[2 * m] = lo(x);
      nk[2 * m + 1] = hi(x);
      dec[2 * m] = lo(d);
      dec[2 * m + 1] = hi(d);
    }
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
    const int ri = g >> 1;
    const float corr = (s_val[t][r0 + ri] - reduce_rows(p, g)) * s_gain[t];

    float2 qn[2 * kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
      const float4 x = *reinterpret_cast<const float4*>(&s_qn[t][32 * m + 4 * g]);
      qn[2 * m] = lo(x);
      qn[2 * m + 1] = hi(x);
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const float c = __shfl_sync(0xffffffffu, corr, (lane & ~7) | (2 * i));
      const float2 c2 = make_float2(c, c);
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < 2 * kChunks; ++j) {
        tile.s[i][j] = __ffma2_rn(c2, nk[j], tile.s[i][j]);
        acc = __ffma2_rn(tile.s[i][j], qn[j], acc);
      }
      p[i] = acc.x + acc.y;
    }
    const float o = reduce_rows(p, g);
    if ((g & 1) == 0) {
      out[(bos + t) * s_out + h * kDim + r0 + ri] = __float2bfloat16(o);
      correction[b * s_corr_row + h * s_corr_head + t * s_corr_pos + r0 + ri] = corr;
    }
  }
}

// One CTA per (request, layer, head) folds the accepted tokens into the state in place:
// S <- S * prod_t d_t + sum_t (u_t k_t) * prod_{s>t} d_s. When prefix caching keeps the state at the
// block boundary, the tokens before it are folded and stored first, then the rest.
template <typename IdxT>
__global__ void __launch_bounds__(kThreads, 2) commit_kernel(
    const int64_t* __restrict__ state_base, const int64_t* __restrict__ state_block_stride,
    const int64_t* __restrict__ corr_base, const int64_t* __restrict__ corr_row_stride,
    const int64_t* __restrict__ kg_base, const int64_t* __restrict__ kg_row_stride,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const IdxT* __restrict__ state_indices, const int* __restrict__ commit_lens,
    const int* __restrict__ final_state_indices, const int* __restrict__ boundary_state_indices,
    const int* __restrict__ boundary_recovery_lens, float lower_bound, bool use_lower_bound,
    int64_t null_block_id, int num_heads, int64_t s_state_head, int64_t s_corr_head,
    int64_t s_corr_pos, int64_t s_kg_head, int64_t s_kg_pos, int64_t s_A_layer, int64_t s_A_head,
    int64_t s_dt_layer, int64_t s_dt_head, int64_t s_idx, bool align) {
  const int b = blockIdx.x, l = blockIdx.y / num_heads, h = blockIdx.y % num_heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane & 7;
  const int r0 = warp * 16 + (lane >> 3) * kRows;

  const int64_t src_idx = int64_t(state_indices[b * s_idx]);
  const int clen = commit_lens[b];
  const int64_t fin_idx = final_state_indices[b];
  if (src_idx <= null_block_id || clen == 0 || fin_idx <= null_block_id) return;
  const int64_t bnd_idx = align ? int64_t(boundary_state_indices[b]) : null_block_id;
  const bool write_boundary = bnd_idx > null_block_id;
  const int blen = write_boundary ? min(boundary_recovery_lens[b], clen) : 0;

  float* state = reinterpret_cast<float*>(state_base[l]) + h * s_state_head + r0 * kDim;
  const int64_t block_stride = state_block_stride[l];
  const float* corr = reinterpret_cast<const float*>(corr_base[l]) + b * corr_row_stride[l] +
                      h * s_corr_head;
  const __nv_bfloat16* kg = reinterpret_cast<const __nv_bfloat16*>(kg_base[l]) +
                            b * kg_row_stride[l] + h * s_kg_head;

  // Within each segment [0, blen) and [blen, clen): w[t] = k_t * prod_{s>t} d_s and the segment's
  // decay prod_t d_t.
  __shared__ __align__(16) float s_w[kMaxQ][kDim];
  __shared__ __align__(16) float s_dec[kMaxQ][kDim];
  __shared__ __align__(16) float s_dseg[2][kDim];
  __shared__ __align__(16) float s_u[kMaxQ][kDim];

  Tile tile;
  tile.load(state + src_idx * block_stride, g);

  const float A = __expf(A_log[l * s_A_layer + h * s_A_head]);
  const float* dtb = dt_bias + l * s_dt_layer + h * s_dt_head;
  if (warp < clen) {
    const int t = warp;
    stage_token(kg + t * s_kg_pos, kg + t * s_kg_pos + kDim, nullptr, dtb, A, lower_bound,
                use_lower_bound, lane, s_w[t], nullptr, s_dec[t]);
    *reinterpret_cast<float4*>(&s_u[t][lane * 4]) =
        *reinterpret_cast<const float4*>(corr + t * s_corr_pos + lane * 4);
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

  auto fold = [&](int t0, int t1, const float* d, int64_t dst_idx) {
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
      const float4 x = *reinterpret_cast<const float4*>(&d[32 * m + 4 * g]);
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        tile.s[i][2 * m] = __fmul2_rn(tile.s[i][2 * m], lo(x));
        tile.s[i][2 * m + 1] = __fmul2_rn(tile.s[i][2 * m + 1], hi(x));
      }
    }
    for (int t = t0; t < t1; ++t) {
      const float4 u = *reinterpret_cast<const float4*>(&s_u[t][r0]);
      const float uu[kRows] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int m = 0; m < kChunks; ++m) {
        const float4 x = *reinterpret_cast<const float4*>(&s_w[t][32 * m + 4 * g]);
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
          const float2 u2 = make_float2(uu[i], uu[i]);
          tile.s[i][2 * m] = __ffma2_rn(u2, lo(x), tile.s[i][2 * m]);
          tile.s[i][2 * m + 1] = __ffma2_rn(u2, hi(x), tile.s[i][2 * m + 1]);
        }
      }
    }
    float* dst = state + dst_idx * block_stride;
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
      for (int m = 0; m < kChunks; ++m) {
        const float2 a = tile.s[i][2 * m], c = tile.s[i][2 * m + 1];
        __stcs(reinterpret_cast<float4*>(dst + i * kDim + 32 * m + 4 * g),
               make_float4(a.x, a.y, c.x, c.y));
      }
  };
  if (write_boundary) fold(0, blen, s_dseg[0], bnd_idx);
  fold(blen, clen, s_dseg[1], fin_idx);
}

}  // namespace

void verify(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor raw_g,
            torch::Tensor raw_beta, torch::Tensor A_log, torch::Tensor dt_bias,
            torch::Tensor state, torch::Tensor correction, torch::Tensor key_gate,
            torch::Tensor out, torch::Tensor query_start_loc, torch::Tensor state_indices,
            double lower_bound, bool use_lower_bound, int64_t null_block_id,
            int64_t spec_query_len) {
  // q, k, raw_g: [tokens, H, K]; v, out: [tokens, H, V]; raw_beta: [tokens, H];
  // state: [blocks, H, V, K]; correction: [reqs, H, Q, V]; key_gate: [reqs, H, Q, 2K].
  const int64_t batch = state_indices.size(0), heads = q.size(1);
  TORCH_CHECK(q.size(2) == kDim && v.size(2) == kDim, "head_dim must be 128");
  TORCH_CHECK(spec_query_len <= kMaxQ, "speculative window too long");
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.stride(3) == 1 && state.stride(2) == kDim,
              "FP32 state with contiguous rows required");
  TORCH_CHECK(state.stride(0) % 4 == 0 && state.stride(1) % 4 == 0, "16-byte aligned state");
  TORCH_CHECK(key_gate.stride(3) == 1 && correction.stride(3) == 1, "contiguous records required");
  for (const auto* t : {&q, &k, &v, &raw_g, &raw_beta, &key_gate, &out})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "bfloat16 activations required");
  for (const auto* t : {&q, &k, &v, &raw_g, &out})
    TORCH_CHECK(t->stride(1) == kDim && t->stride(2) == 1, "contiguous heads required");
  TORCH_CHECK(raw_beta.stride(1) == 1 && A_log.is_contiguous() && dt_bias.is_contiguous() &&
                  A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat,
              "contiguous FP32 gate parameters required");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "int32 query_start_loc required");
  if (batch == 0) return;
  const at::cuda::CUDAGuard guard(q.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 grid(batch, heads);
#define LAUNCH(IdxT)                                                                             \
  verify_kernel<IdxT><<<grid, kThreads, 0, stream>>>(                                            \
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),                                      \
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),                                      \
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                                      \
      reinterpret_cast<const __nv_bfloat16*>(raw_g.data_ptr()),                                  \
      reinterpret_cast<const __nv_bfloat16*>(raw_beta.data_ptr()), A_log.data_ptr<float>(),      \
      dt_bias.data_ptr<float>(), state.data_ptr<float>(), correction.data_ptr<float>(),          \
      reinterpret_cast<__nv_bfloat16*>(key_gate.data_ptr()),                                     \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), query_start_loc.data_ptr<int>(),         \
      state_indices.data_ptr<IdxT>(), float(lower_bound), use_lower_bound, null_block_id,        \
      q.stride(0), k.stride(0), v.stride(0), raw_g.stride(0), raw_beta.stride(0),                \
      state.stride(0), state.stride(1), correction.stride(0), correction.stride(1),              \
      correction.stride(2), key_gate.stride(0), key_gate.stride(1), key_gate.stride(2),          \
      out.stride(0), state_indices.stride(0), int(spec_query_len))
  if (state_indices.scalar_type() == at::kInt) {
    LAUNCH(int32_t);
  } else {
    LAUNCH(int64_t);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void commit(torch::Tensor state_ref, torch::Tensor state_base, torch::Tensor state_block_stride,
            torch::Tensor corr_ref, torch::Tensor corr_base, torch::Tensor corr_row_stride,
            torch::Tensor kg_ref, torch::Tensor kg_base, torch::Tensor kg_row_stride,
            torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state_indices,
            torch::Tensor commit_lens, torch::Tensor final_state_indices,
            torch::Tensor boundary_state_indices, torch::Tensor boundary_recovery_lens,
            double lower_bound, bool use_lower_bound, int64_t null_block_id, int64_t batch,
            bool align) {
  // *_ref: layer 0's tensor, for the strides every layer shares.
  const int64_t layers = state_base.size(0), heads = state_ref.size(1);
  TORCH_CHECK(state_ref.size(2) == kDim && state_ref.size(3) == kDim, "head_dim must be 128");
  TORCH_CHECK(state_ref.scalar_type() == at::kFloat && state_ref.stride(3) == 1 &&
                  state_ref.stride(2) == kDim && state_ref.stride(1) % 4 == 0,
              "FP32 state with contiguous rows required");
  TORCH_CHECK(kg_ref.scalar_type() == at::kBFloat16 && kg_ref.stride(3) == 1 &&
                  corr_ref.stride(3) == 1 && corr_ref.stride(2) % 4 == 0 &&
                  dt_bias.stride(2) == 1,
              "contiguous records required");
  if (batch == 0) return;
  const at::cuda::CUDAGuard guard(state_ref.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  const dim3 grid(batch, layers * heads);
#define LAUNCH(IdxT)                                                                            \
  commit_kernel<IdxT><<<grid, kThreads, 0, stream>>>(                                           \
      state_base.data_ptr<int64_t>(), state_block_stride.data_ptr<int64_t>(),                   \
      corr_base.data_ptr<int64_t>(), corr_row_stride.data_ptr<int64_t>(),                       \
      kg_base.data_ptr<int64_t>(), kg_row_stride.data_ptr<int64_t>(), A_log.data_ptr<float>(),  \
      dt_bias.data_ptr<float>(), state_indices.data_ptr<IdxT>(), commit_lens.data_ptr<int>(),   \
      final_state_indices.data_ptr<int>(), boundary_state_indices.data_ptr<int>(),              \
      boundary_recovery_lens.data_ptr<int>(), float(lower_bound), use_lower_bound,              \
      null_block_id, int(heads), state_ref.stride(1), corr_ref.stride(1), corr_ref.stride(2),   \
      kg_ref.stride(1), kg_ref.stride(2), A_log.stride(0), A_log.stride(1), dt_bias.stride(0),  \
      dt_bias.stride(1), state_indices.stride(0), align)
  if (state_indices.scalar_type() == at::kInt) {
    LAUNCH(int32_t);
  } else {
    LAUNCH(int64_t);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("verify", &verify);
  m.def("commit", &commit);
}
