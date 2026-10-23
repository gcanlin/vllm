# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import numpy as np

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import TokenIds


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_copies_session_id_from_engine_core_request():
    """Test that Request preserves session ID from EngineCoreRequest."""
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session-1",
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.session_id == "session-1"


def test_request_copies_kv_hints_from_engine_core_request():
    """Test that Request preserves KV hints from EngineCoreRequest."""
    kv_hints = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="msg-1",
        actions=[
            KvHintAction(
                action_id="action-1",
                action_type="example.action",
                action_version="1.0",
                payload={},
            )
        ],
    )
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        kv_hints=kv_hints,
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.kv_hints == kv_hints


def test_token_ids_behave_like_a_list():
    ids = TokenIds([1, 2, 3, 4])

    assert ids[1] == 2 and ids[-1] == 4
    assert ids[1:3] == [2, 3] and type(ids[1:3]) is list
    assert type(ids.copy()) is list and ids.copy() == [1, 2, 3, 4]
    expected = [1, 2, 3, 4]
    assert ids == expected and expected == ids and ids == (1, 2, 3, 4)
    assert ids != [1, 2, 3] and (ids != expected) is False
    assert ids + [5] == [1, 2, 3, 4, 5] and [0] + ids == [0, 1, 2, 3, 4]
    assert 3 in ids and ids.index(3) == 2 and list(ids) == [1, 2, 3, 4]

    ids.append(5)
    ids.extend([6, 7])
    ids.extend(TokenIds([8]))
    assert ids.pop() == 8
    del ids[-2:]
    assert ids == [1, 2, 3, 4, 5]

    copied = TokenIds(ids)
    copied.append(6)
    assert ids == [1, 2, 3, 4, 5] and copied == [1, 2, 3, 4, 5, 6]

    restored = pickle.loads(pickle.dumps(ids))
    assert type(restored) is TokenIds and restored == ids
    assert np.asarray(ids).dtype == np.int32
    assert np.asarray(ids).tolist() == [1, 2, 3, 4, 5]


def test_request_stores_token_ids_compactly():
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    request = Request.from_engine_core_request(engine_request, block_hasher=None)
    assert type(request.prompt_token_ids) is TokenIds
    assert type(request._all_token_ids) is TokenIds

    request.append_output_token_ids([4, 5])
    request.append_output_token_ids(6)

    assert request.prompt_token_ids == [1, 2, 3]
    assert request.all_token_ids[:] == [1, 2, 3, 4, 5, 6]
    assert request.all_token_ids.copy() == [1, 2, 3, 4, 5, 6]
    assert request.output_token_ids[:] == [4, 5, 6]
    assert request.num_tokens == 6
