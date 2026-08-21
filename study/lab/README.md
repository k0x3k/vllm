# CPU lab

A GPU-free environment for reading vLLM by running it. Built by
[`setup-cpu-env.sh`](setup-cpu-env.sh); ~2.3 GB of venv, no kernel compilation.

```bash
./study/lab/setup-cpu-env.sh          # once
cd study/lab && ../../.venv/bin/python 01_chunked_prefill.py
```

## What this can and cannot do

**Works.** The whole engine — program #2 in the atlas. Real tokenizers, real
`Request` objects, the real `Scheduler`, `KVCacheManager` and `BlockPool`, real
prefix-cache fingerprinting, real preemption. Every pure-Python test under
`tests/v1/core` passes. `breakpoint()` anywhere and step.

**Does not work.** `LLM.generate()` and anything that runs a forward pass. There
are no compiled kernels, so there is no worker. The scripts here substitute a
`fake_model_output` that emits one token per running request — the same trick
`tests/v1/core/test_scheduler.py` uses.

The engine never touches the GPU, which is exactly why it can be studied without
one.

## Scripts

| Script | Shows | Atlas |
|---|---|---|
| [`01_chunked_prefill.py`](01_chunked_prefill.py) | One 122-token prompt sliced into 32-token rounds, then 1-token decodes | §4 |
| [`02_prefix_cache.py`](02_prefix_cache.py) | A 90-token shared prefix scoring an 80-token cache hit — block-aligned, so 5 full blocks of 16 | §5 |
| [`03_free_list.py`](03_free_list.py) | Fingerprinted blocks landing at the back of the free list; LRU with no eviction code | §5 |

[`_lab.py`](_lab.py) holds the shared setup and reuses `create_scheduler` from
`tests/v1/core/utils.py` rather than reinventing it.

## Two things worth knowing

`VLLM_TARGET_DEVICE` is read at **runtime** by `cpu_platform_plugin()`, not only
at build time. Without it the platform resolves to `UnspecifiedPlatform`,
`DeviceConfig` cannot infer a device, and building a `VllmConfig` fails.
`_lab.py` sets it before importing vllm.

`tests/v1/core/conftest.py` overrides `should_do_global_cleanup_after_test`.
The base fixture tears down a distributed environment that CPU-only unit tests
never set up; the base docstring sanctions this override and it makes the suite
~10x faster (3.55s → 0.08s).

## The exercise

Predict, then break, then verify. In `BlockPool.free_blocks`
(`vllm/v1/core/block_pool.py:719`):

1. Swap `blocks_to_evict_first` and `blocks_to_evict_last`.
2. Predict what `03_free_list.py` prints and which of the six
   `free_kv_cache_block_queue` tests fail — *before* running anything.
3. Run `../../.venv/bin/python 03_free_list.py` and
   `.venv/bin/python -m pytest tests/v1/core/test_kv_cache_utils.py -k free_kv_cache_block_queue`.

The gap between prediction and result is the part not yet understood. Then make
`remove()` unlink only the forward pointer and see whether any test catches it —
if none does, that is a real coverage gap.
