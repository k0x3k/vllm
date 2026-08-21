"""Shared setup for the lab scripts.

Import this first: it forces the CPU platform before vllm is imported and puts
the repo's test helpers (tests/v1/core/utils.py) on the path so the lab reuses
the same scaffolding the real test suite uses.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from v1.core.utils import create_requests, create_scheduler  # noqa: E402


def tokenize(text: str, model: str = "facebook/opt-125m") -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    return tok(text).input_ids


def make_request(
    req_id: str,
    token_ids: list[int],
    max_tokens: int,
    block_size: int = 16,
    cache_salt: str | None = None,
):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    return Request(
        request_id=req_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def fake_model_output(scheduler, token_id: int = 1000):
    """Stand in for the worker.

    A request still working through its prompt produces no token this round —
    only one whose prompt is fully computed gets sampled. Getting this wrong
    silently inflates the token counts, so it is worth mirroring faithfully.
    """
    from vllm.v1.outputs import ModelRunnerOutput

    running = scheduler.running
    return ModelRunnerOutput(
        req_ids=[r.request_id for r in running],
        req_id_to_index={r.request_id: i for i, r in enumerate(running)},
        sampled_token_ids=[
            [token_id] if r.num_computed_tokens >= r.num_prompt_tokens else []
            for r in running
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
