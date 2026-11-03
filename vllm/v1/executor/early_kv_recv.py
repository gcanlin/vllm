# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report completed KV pulls between model outputs and the next schedule()."""

import os
import threading
import time
from typing import TYPE_CHECKING

import torch
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.utils import sched_yield
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

logger = init_logger(__name__)

# Interval between polls while the worker waits for its next RPC.
_POLL_INTERVAL_S = 0.0002
# Without a step RPC for this long the engine is idle or paused: stop polling
# and let the worker block in dequeue().
_ACTIVE_WINDOW_S = 1.0
# The notified thread also re-checks this often, should a notification be lost.
_NOTIFIED_IDLE_S = 0.05


class EarlyKVOutputAggregator(KVOutputAggregator):
    """Also counts receives that workers report between step outputs."""

    def add_finished_recving(self, req_ids: set[str]) -> set[str]:
        """Count one worker's report and return the requests now finished.

        Each worker reports a load once, so votes are not deduplicated.
        """
        failed = req_ids & self._failed_recving_pending
        if failed:
            # Let normal aggregation count these votes and deliver the failure.
            self.ft_defer_kv_connector_outputs(
                [KVConnectorOutput(finished_recving=failed)]
            )
        finished = set[str]()
        expected = self._expected_finished_count
        for req_id in req_ids - failed:
            remaining = self._recv_remaining_count.get(req_id, expected) - 1
            if remaining:
                self._recv_remaining_count[req_id] = remaining
            else:
                finished.add(req_id)
                self._recv_remaining_count.pop(req_id, None)
        return finished


class EarlyKVRecvReceiver:
    """Engine end: receives the workers' reports."""

    def __init__(self) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)

    def drain(self) -> list[set[str]]:
        """Pending reports, one per worker poll; each is a separate vote."""
        notifications: list[set[str]] = []
        while True:
            try:
                notifications.append(self._socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                return notifications

    def close(self) -> None:
        self._socket.close(linger=0)


class EarlyKVRecvSender:
    """Worker end: reports completed receives to the engine.

    If the connector calls a notifier as receives complete
    (set_recv_done_notifier), a thread woken by it reports them at once, also
    while the worker thread is inside an RPC. Otherwise the worker thread
    polls the connector between RPCs.
    """

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        self._last_step_rpc = time.perf_counter()
        self._next_poll = 0.0
        # Serializes polls (and the PUSH socket) between the worker thread and
        # the notified thread.
        self._poll_lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._thread_checked = False

    def step_rpc_done(self) -> None:
        """Worker thread: an ``execute_model`` or ``sample_tokens`` RPC finished."""
        self._last_step_rpc = time.perf_counter()
        if not self._thread_checked:
            # The connector exists by the first step RPC.
            self._thread_checked = True
            self._start_notified_thread()

    def _start_notified_thread(self) -> None:
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        if not connector.set_recv_done_notifier(self._wake.set):
            return
        self._thread = threading.Thread(
            target=self._notified_loop,
            args=(connector,),
            name="early-kv-recv",
            daemon=True,
        )
        self._thread.start()

    def _notified_loop(self, connector: "KVConnectorBase_V1") -> None:
        while not self._stopping:
            self._wake.wait(_NOTIFIED_IDLE_S)
            # Cleared before polling: a receive finishing during the poll sets
            # it again and is picked up on the next pass.
            self._wake.clear()
            if not self._stopping and connector.has_pending_recving():
                self.poll(connector)

    def wait_for_poll(self, rpc_mq: MessageQueue) -> bool:
        """Spin until the next poll is due, yielding as dequeue() does.

        Returns False once an RPC is queued, or at once when no step RPC
        arrived recently or no receive is pending.
        """
        if (
            self._thread is not None
            or time.perf_counter() - self._last_step_rpc > _ACTIVE_WINDOW_S
            or not has_kv_transfer_group()
            or not get_kv_transfer_group().has_pending_recving()
        ):
            return False
        while not (rpc_mq.shutting_down or rpc_mq.readable()):
            if time.perf_counter() >= self._next_poll:
                return True
            sched_yield()
        return False

    def poll(self, connector: "KVConnectorBase_V1 | None" = None) -> None:
        """Send completed receives; let the worker monitor handle fatal errors.

        The notified thread passes the connector it started with, since the
        worker thread unsets the global one around dummy runs.
        """
        try:
            with self._poll_lock, torch.inference_mode():
                if connector is None:
                    connector = get_kv_transfer_group()
                req_ids = connector.poll_finished_recving()
                if req_ids:
                    self._socket.send_pyobj(req_ids)
        except Exception:
            logger.exception("Early KV receive poll failed")
            # Wake the monitor even while the engine is paused. Receive state
            # may already be lost; neither a later RPC nor a retry can recover it.
            os._exit(1)
        finally:
            # Measured from the end of a poll, so long polls do not run back
            # to back.
            self._next_poll = time.perf_counter() + _POLL_INTERVAL_S

    def close(self) -> None:
        if self._thread is not None:
            self._stopping = True
            self._wake.set()
            self._thread.join()
            self._thread = None
        self._socket.close(linger=0)
