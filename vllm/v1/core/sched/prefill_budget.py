# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill step budgets of a data-parallel prefill instance.

The DP ranks of an instance run each step in lockstep, so a rank that would run
much longer than the others stretches the step for all of them. The budgets
here bound and balance each rank's prefill work. The ranks share their state
through a small array in /dev/shm instead of a blocking collective, so ranks on
other hosts are not seen.
"""

import math
import os
import time

import numpy as np

from vllm.config import VllmConfig
from vllm.config.scheduler import PrefillStepBudgetConfig, PrefillTokenCapConfig

# Weight of the latest step in the context demand a rank publishes.
_CTX_DEMAND_WEIGHT = 0.3


def _shared_array(vllm_config: VllmConfig, name: str, size: int) -> np.ndarray:
    """A float64 array shared by the instance's DP ranks on this host."""
    parallel_config = vllm_config.parallel_config
    path = (
        f"/dev/shm/vllm_{name}_{parallel_config.data_parallel_rpc_port}"
        f"_{parallel_config.data_parallel_size}"
    )
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if os.fstat(fd).st_size != size * 8:
            os.ftruncate(fd, size * 8)
    finally:
        os.close(fd)
    return np.memmap(path, dtype=np.float64, mode="r+", shape=(size,))


def _dp_rank(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    return parallel_config.data_parallel_rank % parallel_config.data_parallel_size


class PrefillStepBudget:
    """Cuts prefill chunks so that a step fits its predicted-time budgets.

    Per step: begin_step(), then fit() before and add() after scheduling each
    prefill chunk, then end_step().
    """

    def __init__(
        self,
        config: PrefillStepBudgetConfig,
        vllm_config: VllmConfig,
        block_size: int,
    ) -> None:
        self.config = config
        self.block_size = block_size
        self._rank = _dp_rank(vllm_config)
        size = vllm_config.parallel_config.data_parallel_size
        self._ctx_demands: np.ndarray | None = None
        if config.ctx_balance is not None:
            self._ctx_demands = _shared_array(vllm_config, "prefill_ctx_demand", size)
            self._ctx_demands[self._rank] = config.ctx_balance[2] / 2
        self._step_demands: np.ndarray | None = None
        if config.dp_budget is not None:
            self._step_demands = _shared_array(vllm_config, "prefill_step_demand", size)
            self._step_demands[self._rank] = config.dp_budget[2]
        self.begin_step()

    def _cost(self, num_new: int, num_computed: int) -> float:
        cfg = self.config
        return (
            cfg.d
            + cfg.g * num_computed
            + num_new * (cfg.a + cfg.b * num_computed)
            + cfg.b2 * num_new * num_new
        )

    def _ctx_cost(self, num_new: int, num_computed: int) -> float:
        cfg = self.config
        return cfg.g * num_computed + cfg.b * num_new * num_computed

    def _max_tokens(self, num_computed: int) -> float:
        """New tokens of a chunk at num_computed that fit the step time left."""
        cfg = self.config
        rest = self._cost_left - cfg.d - cfg.g * num_computed
        if rest <= 0:
            return 0
        if rest == math.inf:
            return math.inf
        per_token = cfg.a + cfg.b * num_computed
        if cfg.b2 == 0:
            return rest // per_token
        return (
            (-per_token + (per_token * per_token + 4 * cfg.b2 * rest) ** 0.5)
            / (2 * cfg.b2)
        ) // 1

    def _max_ctx_tokens(self, num_computed: int) -> float:
        """New tokens of a chunk at num_computed that fit the context time left."""
        cfg = self.config
        rest = self._ctx_left - cfg.g * num_computed
        if rest <= 0:
            return 0
        if cfg.b == 0 or num_computed == 0:
            return math.inf
        return rest / (cfg.b * num_computed)

    def begin_step(self) -> None:
        cfg = self.config
        self._cost_left = cfg.budget_ms - cfg.e
        if self._step_demands is not None:
            assert cfg.dp_budget is not None
            factor, floor_ms, cap_ms = cfg.dp_budget
            others = np.delete(self._step_demands, self._rank)
            peak = float(others.max()) if others.size else cap_ms
            budget = min(cap_ms, max(floor_ms, factor * peak))
            self._cost_left = min(self._cost_left, budget - cfg.e)
        if self._ctx_demands is not None:
            assert cfg.ctx_balance is not None
            factor, floor_ms, cap_ms = cfg.ctx_balance
            demands = self._ctx_demands[self._ctx_demands > 0]
            mean = float(demands.mean()) if demands.size else cap_ms
            self._ctx_left = min(cap_ms, max(floor_ms, factor * mean))
        else:
            self._ctx_left = cfg.ctx_budget_ms
        self._cost_start = self._cost_left
        self._ctx_start = self._ctx_left
        # Predicted time of the chunk parts that were cut off this step.
        self._cost_excess = self._ctx_excess = 0.0
        self._num_chunks = 0
        self._num_skips = 0

    def fit(self, num_computed: int, num_new: int) -> int:
        """The longest prefix of a chunk that fits the step; the step's first
        chunk keeps at least a block."""
        fit = int(
            min(
                num_new,
                self._max_tokens(num_computed),
                self._max_ctx_tokens(num_computed),
            )
        )
        if self._num_chunks == 0:
            fit = max(fit, min(num_new, self.block_size))
        if fit < num_new:
            rest, computed = num_new - fit, num_computed + fit
            self._cost_excess += self._cost(rest, computed) - self.config.d
            self._ctx_excess += self._ctx_cost(rest, computed)
        return fit

    def add(self, num_computed: int, num_new: int) -> None:
        """Charge a scheduled prefill chunk to the step."""
        self._cost_left -= self._cost(num_new, num_computed)
        self._ctx_left -= self._ctx_cost(num_new, num_computed)
        self._num_chunks += 1

    def may_skip(self, num_computed: int) -> bool:
        """Whether to scan past a waiting request that fit only the step time:
        shorter-context requests behind it may still fit the context budget."""
        if (
            self._num_skips < self.config.skip_scan
            and self._max_tokens(num_computed) > 0
        ):
            self._num_skips += 1
            return True
        return False

    def end_step(self) -> None:
        """Publish this rank's demand to the instance's other ranks."""
        if self._ctx_demands is not None and self._num_chunks:
            assert self.config.ctx_balance is not None
            demand = min(
                self._ctx_start - self._ctx_left + self._ctx_excess,
                self.config.ctx_balance[2],
            )
            self._ctx_demands[self._rank] = (
                1 - _CTX_DEMAND_WEIGHT
            ) * self._ctx_demands[self._rank] + _CTX_DEMAND_WEIGHT * demand
        if self._step_demands is not None:
            # An idle rank publishes 0, so that a lone busy rank is held to the
            # floor rather than to the others' stale demand.
            busy = self._num_chunks or self._cost_excess
            self._step_demands[self._rank] = (
                self.config.e + self._cost_start - self._cost_left + self._cost_excess
                if busy
                else 0.0
            )


class PrefillTokenCap:
    """Caps prefill steps while the instance's queues are short."""

    def __init__(self, config: PrefillTokenCapConfig, vllm_config: VllmConfig):
        self.config = config
        self._rank = _dp_rank(vllm_config)
        self._size = vllm_config.parallel_config.data_parallel_size
        # [0, size): each rank's waiting requests; [size]: 1 capped, 0 full;
        # [size + 1]: when a rank last waited for more than low_waiting.
        self._state = _shared_array(vllm_config, "prefill_token_cap", self._size + 2)
        self._state[self._rank] = 0
        if self._rank == 0:
            self._state[self._size] = 1.0
            self._state[self._size + 1] = 0.0

    def capped(self, num_waiting: int) -> bool:
        """Publish this rank's waiting requests; return whether the step is
        capped."""
        state, size, cfg = self._state, self._size, self.config
        state[self._rank] = num_waiting
        peak = float(state[:size].max())
        now = time.monotonic()
        if peak > cfg.low_waiting:
            state[size + 1] = now
        if state[size] and peak > cfg.max_waiting:
            state[size] = 0.0
        elif not state[size] and now - float(state[size + 1]) >= cfg.hold_s:
            state[size] = 1.0
        return bool(state[size])
