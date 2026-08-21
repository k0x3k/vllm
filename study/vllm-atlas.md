# How vLLM turns a prompt into tokens

*Field guide · vllm-project/vllm · main @ 4b7cb94 · 2026-08-20*

A map of the vLLM code: which programs actually run, how a request travels through them, and what each part is for. Written against your checkout at `~/lab/code/infer/vllm`, with hands-on exercises for the RTX 5090 in this machine. Read it top to bottom once, then keep it open as a reference.

Reference machine: host `gl1tchx` · GPU `RTX 5090 (SM 12.0)` · editable install with prebuilt kernels · torch `2.13.0+cu130` · python `3.12` (uv venv)

## §1 What's in the repo

vLLM is mostly Python — that's what you'll read and change day to day. The heavy GPU math lives in CUDA C++ and in a few Python-based languages that generate GPU code (Triton and friends). There's also an optional web server written in Rust. Rough proportions:

Python ×4,279 files · Rust ×306 · CUDA/C++ ×217 (~86% / 8% / 6%)

### The folders that matter

- `v1/` — the engine itself: the scheduler, the memory manager for past tokens, the workers that run the model, the sampler. Most of the action is here.
- `entrypoints/` — the ways in: the `LLM` Python class, the OpenAI-style web server, the command line.
- `engine/` — mostly old names kept alive for compatibility, plus `EngineArgs`, which turns command-line flags into config objects.
- `renderers/` — turns your text (or a chat conversation) into token numbers the model understands.
- `model_executor/` — the model definitions, their building-block layers, weight loading, and quantization (running models at lower precision to save memory).
- `models/` — a newer home for a handful of models that ship different code per GPU vendor (DeepSeek V3.2/V4, Kimi K3, …).
- `config/` — `VllmConfig` and its ~25 sub-configs. Every command-line flag is generated automatically from these.
- `compilation/` — makes the model faster by compiling it ahead of time (§8).
- `ir/` + `kernels/` — a small system that lets one operation (like a layer norm) have several competing GPU implementations, picking the best one available.
- `distributed/` — everything about running on more than one GPU or machine.
- `csrc/` — the CUDA C++ source. Compiles into `_C_stable_libtorch.so`.
- `rust/` — the optional Rust web server, `vllm-rs`.
- `platforms/` — one file per hardware type (NVIDIA, AMD, Intel, CPU) holding all the "on this hardware, do it this way" decisions.
- `parser/`, `tool_parsers/`, `reasoning/` — reads the model's output as it streams, picking out tool calls and reasoning sections.
- `multimodal/`, `lora/` — images/audio in prompts, and LoRA adapters (small add-on weights).

> **Your install** — Your setup is an editable install with prebuilt GPU code. In plain terms: **any Python file you edit takes effect the next time you run vLLM** — no build step. The compiled GPU parts came ready-made from a download. Only if you change files under `csrc/` do you need a real build (`uv pip install -e . --no-build-isolation`; the guide is `docs/contributing/incremental_build.md`).

## §2 The three programs

A running vLLM server is really three separate programs talking to each other. Almost every file in the repo belongs to exactly one of them, so knowing "which program am I in?" is the most useful habit you can build while reading.

- **The front end** — takes requests in, turns text into token numbers, and turns the model's token numbers back into text for the reply.
- **The engine** — the decision maker. Each round, it picks which requests get to run and hands out memory. It never touches the GPU.
- **The workers** — one per GPU. They hold the model weights and do all the actual computation.

> **Diagram** — Three boxes: the front end holds the AsyncLLM class and the text-to-tokens and tokens-to-text steps. It sends requests over a socket to the engine, which holds the scheduler and the memory manager. The engine sends work over a shared-memory queue to the workers, which hold the model and do the GPU math.
>
> *The three programs. Front end and engine talk over a socket; engine and workers share a chunk of memory for speed. The scheduler makes every decision but never does GPU work itself.*

| Program | Starts at | Holds |
|---|---|---|
| Front end | `vllm/entrypoints/launchers/api_server/entry.py` | The web app, `AsyncLLM`, text→tokens (`InputProcessor`), tokens→text (`OutputProcessor`) |
| Engine | `EngineCoreProc.run_engine_core` · `vllm/v1/engine/core.py:1271` | `Scheduler`, `KVCacheManager`, plus two helper threads that handle the sockets |
| Worker (one per GPU) | `WorkerProc.worker_main` · `vllm/v1/executor/multiproc_executor.py:853` | `Worker → GPUModelRunner`, the weights, the cache memory, the sampler |

There's an optional fourth program: the **Rust front end** (`vllm-rs`, turned on with `VLLM_USE_RUST_FRONTEND=1`). It replaces the Python front end entirely — Rust handles the web traffic and talks to the same engine over the same socket protocol. It's a full rewrite of that layer, not a wrapper around the Python one.

## §3 A request, start to finish

Here is the whole journey of one request, as an ordered list of the functions it passes through. This is the offline path (`LLM.generate()`). The web-server path only differs at the top — request comes in over HTTP, the chat template is applied — and joins at step 3. Copper rings mark the big hand-offs; dashed lines mark where we cross from one program into another.

1. **`LLM.generate(prompts, sampling_params)`**
   `vllm/entrypoints/llm.py:418`
   Checks the arguments, fills in defaults, passes the work along.

2. **`renderer.render_cmpl(…) → LLMEngine.add_request(…)`**
   `vllm/entrypoints/offline_utils.py:290 · vllm/v1/engine/llm_engine.py:218`
   Your text becomes token numbers here. Rendering happens lazily, so tokenizing and running the model overlap.

3. **`InputProcessor.process_inputs(…)`**
   `vllm/v1/engine/input_processor.py:281`
   Final checks (length limits, image preprocessing), then the request is packed into a compact message for the engine. If you asked for `n>1` answers, it becomes n separate requests right here.

   ↓ **CROSSING INTO THE ENGINE · OVER A SOCKET**

4. **`EngineCore.preprocess_add_request(…)`**
   `vllm/v1/engine/core.py:968 — runs on a helper thread`
   Builds the engine's `Request` object. The fingerprints used for cache reuse (§5) are computed here, on the helper thread, so the main loop never waits for them.

5. **`Scheduler.add_request(…)`**
   `vllm/v1/core/sched/scheduler.py:2305`
   The request joins the waiting line.

6. **`EngineCoreProc.run_busy_loop → EngineCore.step()`**
   `vllm/v1/engine/core.py:1391 · core.py:583`
   The heartbeat of the whole system. Each beat: decide → run the model → pick tokens → record results. A nice trick hides here: while the GPU is busy with the forward pass, the engine prepares the "which tokens are allowed" mask for structured output — two things at once.

7. **`Scheduler.schedule()`**
   `vllm/v1/core/sched/scheduler.py:477`
   The decision: who runs this round, how many tokens each request gets, which memory blocks they use, and whether anyone has to be kicked out. Details in §4.

   ↓ **CROSSING INTO THE WORKERS · OVER SHARED MEMORY**

8. **`MultiprocExecutor.execute_model(…)`**
   `vllm/v1/executor/multiproc_executor.py:111`
   Broadcasts "here's the plan" to every worker at once. Only one worker sends results back — the rest hold pieces of the same answer (see §9).

9. **`GPUModelRunner.execute_model → _sample`**
   `vllm/v1/worker/gpu_model_runner.py:4288 · :3759`
   The GPU does its thing: update the batch, run the model once over all scheduled requests together, then pick the next token for each.

10. **`Scheduler.update_from_output(…)`**
   `vllm/v1/core/sched/scheduler.py:1737`
   Bookkeeping: append the new tokens, check who's finished (hit a stop token, or their length limit), and package the results for the front end.

   ↓ **BACK TO THE FRONT END · OVER A SOCKET**

11. **`OutputProcessor.process_outputs(…)`**
   `vllm/v1/engine/output_processor.py:598`
   Token numbers become text again. Stop *strings* (as opposed to stop tokens) can only be spotted after this conversion — that's why it's the front end that then tells the engine "you can drop this one".

12. **`Your code gets the result`**
   `vllm/v1/engine/async_llm.py:550 · vllm/entrypoints/offline_utils.py:573`
   The server streams each piece to the client as it arrives; the offline path just loops until everyone is done and returns the list.

> **Reading order** — If you only read three files end to end, make them `vllm/v1/engine/core.py` (the loop), `vllm/v1/core/sched/scheduler.py` (the decisions), and `vllm/v1/worker/gpu_model_runner.py` (the execution). Everything else hangs off these three.

## §4 The scheduler: one loop, no special cases

Background, in one paragraph: when a model reads your prompt, it processes many tokens at once (people call this *prefill*). When it writes its answer, it produces one token at a time (*decode*). Most serving systems treat these as two different modes. vLLM's scheduler deliberately doesn't — a comment in the source says so outright. Every request just carries two numbers: how many tokens are *done*, and how many it *needs*. Each round, the scheduler shares out a fixed budget of tokens to close those gaps.

> **Diagram** — Three requests drawn as progress bars. Request A is partway through reading its prompt, with a big gap left. Request B is writing its answer, with a gap of one token. Request C has a few guessed tokens waiting to be checked. One shared token budget gets split across all three in the same round.
>
> *Long prompts don't get a special "prefill mode" — a prompt that doesn't fit this round's budget simply continues next round ("chunked prefill" is just this clipping). A decode request is one whose gap happens to be a single token.*

### The two passes inside `schedule()`

**Pass 1 — requests already running come first.** They get budget before anyone new is let in. This is why answers keep flowing smoothly even when big prompts arrive: a request that's mid-answer never loses its place to a newcomer. If one of them can't get the memory it needs, the scheduler kicks someone out (see below) and tries again.

**Pass 2 — the waiting line.** Skipped entirely if anyone was kicked out this round. For each waiting request: check whether the start of its prompt is already cached (§5), clip its ask to the remaining budget, grab memory, admit. A request that's waiting on something else — say its output-format rules are still being prepared — steps aside instead of blocking the line.

> **Kicking a request out = redoing its work** — When memory runs out, the scheduler frees *everything* a victim request had and sends it back to the front of the waiting line with its progress counter reset to zero (`scheduler.py:1340`). There's no "move it to CPU memory" fallback in v1. It's cheaper than it sounds: the fingerprints of its old blocks still exist, so when it comes back it usually finds most of its work still cached and skips ahead.

One subtlety worth knowing early: the scheduler marks tokens as "done" *the moment it schedules them*, before the GPU has actually run — so the next round can be planned without waiting. If some of those tokens turn out to be rejected guesses (speculative decoding), the count is corrected afterwards in `update_from_output`.

## §5 Memory for past tokens

Background: for every token a model has seen, it keeps a small stash of working data (the "KV cache") so it doesn't have to reread the whole conversation for each new token. This stash is big — it's usually what limits how many requests fit on a GPU. vLLM's trick (the original "PagedAttention" idea) is to slice this memory into fixed blocks of 16 tokens and hand blocks out like a memory allocator, instead of reserving one big slab per request.

The blocks live in one shared pool (`vllm/v1/core/block_pool.py`). On top of the pool sits a manager that handles models mixing different attention styles (some layers look at everything, some only at a recent window) — each style keeps and throws away blocks by different rules.

### Reusing work between requests

Two requests that start with the same text — same system prompt, say — do exactly the same work for those tokens. To skip the repeat, every full block gets a fingerprint. The clever part is that each fingerprint mixes in the previous block's fingerprint:

> **Diagram** — Four memory blocks in a row. Each block's fingerprint is computed from the previous block's fingerprint plus its own tokens, chained left to right. So a single match on block 3 proves the entire start of the prompt matches, and a miss anywhere means everything after it misses too.
>
> *Prefix caching. The fingerprint math is `hash_block_tokens` in `vllm/v1/core/kv_cache_utils.py:618`, and it runs on a helper thread so it never slows the main loop.*

Three design choices explain most of the code in this area:

- **There is no separate eviction algorithm.** Freed blocks that have a fingerprint go to the *back* of the free list, blocks without one go to the *front*. Taking from the front then naturally throws away the least-recently-used cached data. The whole policy is list ordering (`block_pool.py:719`).
- **The free list is built for grabbing from the middle.** When a new request wants a cached block, it's plucked straight out of the free list in one step — the list is doubly linked precisely for this.
- **Guessed tokens are never cached.** Speculative decoding runs on unverified guesses; only tokens that actually made it into an answer get fingerprinted.

## §6 Running the model

### What one round looks like inside the worker

1. `_update_states` — sync the standing batch. The worker keeps every active request's data (sampling settings, token buffers) parked in big reusable GPU arrays, so each round only applies the diff: who joined, who left, who got more tokens.
2. `_prepare_inputs` — lay out this round's token IDs, positions, and where each request's memory blocks live.
3. `_build_attention_metadata` — the per-round instructions the attention kernels need.
4. `_preprocess` — if there are images/audio, run their encoder and splice the results into the input.
5. The forward pass — one run of the model over *all* scheduled requests at once. Attention layers don't get their instructions as function arguments; they read them from a per-round shared object (`ForwardContext`, `vllm/forward_context.py`). That indirection is what lets the surrounding code be compiled and replayed (§8).
6. `_sample` — pick each request's next token from the model's output scores.

### How a model class gets picked, built, and filled

1. **`ModelRegistry.resolve_model_cls(architectures)`**
   `vllm/model_executor/models/registry.py`
   A name-to-class lookup ("LlamaForCausalLM" → the Llama file). Checking a model's abilities happens in a throwaway child process, so merely importing model code can't accidentally grab the GPU. If nothing matches, it falls back to running the model through HuggingFace Transformers.

2. **`initialize_model(vllm_config, …)`**
   `vllm/model_executor/model_loader/utils.py:38`
   Builds the layers. This is also where quantization hooks in: every linear layer asks the quant config "how should I store and multiply my weights?" and gets a method object back.

3. **`loader.load_weights → model.load_weights(…)`**
   `vllm/model_executor/model_loader/default_loader.py`
   Streams the checkpoint files in. Each parameter knows how to load itself — including taking only its shard when the model is split across GPUs, and fusing separate q/k/v files into one combined weight. A final pass lets quantized layers repack into their fast layout.

### Choosing the next token

The sampler (`vllm/v1/sample/sampler.py`) is short and readable, and the *order* of its steps is the contract: first the hard bans (forbidden tokens, banned words), then the score adjustments (repetition penalties and friends), then temperature, then top-k/top-p trimming, then the actual pick. When speculative decoding is on, a different component (the rejection sampler) takes over: it checks the cheap model's guesses against the real model and keeps the ones that hold up.

## §7 What runs on *your* GPU

vLLM has many interchangeable attention implementations ("backends"). At startup it walks a ranked list for your hardware and picks the first one whose requirements all pass (`vllm/platforms/cuda.py`). For your 5090 — compute capability 12.0, "consumer Blackwell" — here's how that plays out today:

| Situation | What you get | Worth knowing |
|---|---|---|
| Normal attention | `FLASH_ATTN` — FlashAttention **2** | Versions 3 and 4 are only enabled for the server chips (H100/B200 class). Consumer Blackwell stays on version 2. Next in line: `FLASHINFER`, then `TRITON_ATTN`. |
| MLA (DeepSeek-style attention) | `TRITON_MLA` | The only MLA implementation that works on your chip — the fast ones are all server-chip-only. There's one SM120-specific extra: `FLASHINFER_MLA_SPARSE_SM120`. |
| Built for your chip | Marlin (quantized weights), the FP8 and NVFP4 matrix-multiply kernels for SM120, DeepGEMM, QuTLASS, FlashKDA. | |
| *Not* built for your chip | Machete, FlashMLA, CUTLASS MLA, W4A8, hadacore, AllSpark. PRs touching these can't be tested on this machine. | |

> **Why this is your opening** — Most vLLM maintainers develop on H100s and B200s, so the consumer-Blackwell column gets less attention — the table above has visible gaps (stuck on FlashAttention 2, only one MLA option). Making kernels work on SM120, tuning them, and adding SM120 test coverage are contributions this exact machine is unusually well placed to make.

### Where the GPU code actually lives

- **`csrc/libtorch_stable/`** — the CUDA C++ kernels. They compile into `_C_stable_libtorch.so` but show up in Python as `torch.ops._C.*`; the hand-written wrappers are in `vllm/_custom_ops.py`.
- **The famous 2023 "paged attention" kernel is gone.** The kernel from the original paper no longer exists in the CUDA tree — that job is now done by FlashAttention, FlashInfer, or Triton kernels. Blog posts describing it are history, not the current code.
- **GPU code written in Python** — a lot of kernels are written in Triton (Python that compiles to GPU code), and newer ones in CuTe DSL (NVIDIA's Python kernel language — there's a full FlashAttention-4 written in it at `vllm/vllm_flash_attn/cute/flash_fwd_sm120.py`, including an SM120 file) and a couple of others (Helion, TileLang).
- **`vllm/ir/`** — a small "one op, many implementations" registry. An op like `rms_norm` has a plain PyTorch version plus faster registered alternatives, and the compiler picks the best available one. New fusable ops are expected to follow this pattern.
- **Outside dependencies** — FlashInfer comes from pip; FlashAttention, DeepGEMM, FlashMLA and friends are fetched and built by CMake. Your prebuilt install skipped all of that.

## §8 How the model gets compiled

Two facts to hold: **torch.compile** traces Python once and generates fast fused code for it, and **CUDA graphs** record a whole sequence of GPU launches so it can be replayed later with almost zero launch cost. Both want things to be predictable — same shapes, same steps — and attention is the one part of the model that isn't: how much history each request looks at changes every round.

vLLM's answer: trace the model once, then **cut the traced graph at every attention call**. The pieces between the cuts are predictable, so each piece gets compiled and recorded as a CUDA graph. Attention itself runs normally in the gaps, reading its per-round instructions from the shared context object from §6. Best of both: fused, replayable code around eager, flexible attention.

> **Diagram** — A five-stage pipeline. The model's forward function is traced once, cut into pieces at every attention call, each piece is compiled and fusion passes run on it, and each compiled piece gets recorded as a replayable CUDA graph per batch size.
>
> *The default pipeline. "Fusion passes" merge neighboring operations (layer-norm + quantize, activation + quantize, and so on) into single kernels while each piece compiles.*

- **How much of this you get** is set by `--optimization-level` (O0 = none of it, O2 = the default: everything above, O3 = a bit more).
- **Where it's cached**: `~/.cache/vllm/torch_compile_cache/…`. Each compile also dumps a readable `computation_graph.py` showing exactly where the cuts landed — exercise 5 has you open it.
- **First startup is slow, later startups are fast** — that's this cache doing its job. Delete it if you suspect it's stale.

## §9 Many GPUs, one table

On your single-GPU machine all of this collapses to "one worker, no splitting" — but the names show up everywhere in the code, so here's what each one means and who's in charge of it.

| Name | Plain meaning | Run from | Where to look |
|---|---|---|---|
| TP — tensor parallel | Split every weight matrix across GPUs; all GPUs work on every token together | Inside the layers themselves — they call the collective operations directly | `model_executor/layers/linear.py` |
| PP — pipeline parallel | Split the model by layers: GPU 1 has the first half, GPU 2 the second; results flow between them | The worker passes partial results to the next stage | `v1/worker/gpu_worker.py:1100` |
| DP — data parallel | Full copies of the model, each serving different requests | One engine per copy, plus a small coordinator process | `v1/engine/core.py`, `coordinator.py` |
| EP — expert parallel | For mixture-of-experts models: spread the experts across GPUs and route tokens to whichever GPU has their expert | The communication layer, with a live rebalancer (EPLB) shuffling experts to even out load | `distributed/device_communicators/all2all.py`, `distributed/eplb/` |
| Disaggregated prefill | Separate machines for reading prompts vs. writing answers, shipping the cache between them | "KV connectors", each split into a scheduler half and a worker half | `distributed/kv_transfer/kv_connector/v1/` |

## §10 Exercises

Ordered from light to deep. All assume `cd ~/lab/code/infer/vllm && source .venv/bin/activate`. Because your install is editable, any Python edit is live on the next run — no rebuilding, ever. `Qwen/Qwen3-0.6B` is already downloaded and loads in seconds.

### 1 · Squash it into one process and step through  *(warm-up)*

One environment variable folds all three programs into a single debuggable process:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-0.6B', max_model_len=2048, gpu_memory_utilization=0.5)
print(llm.generate(['Hello'], SamplingParams(max_tokens=8))[0].outputs[0].text)"
```

**Then:** Put `breakpoint()` at the top of `EngineCore.step` (`vllm/v1/engine/core.py:583`) and run again. Walk through one full round: decide → run → pick tokens → record. One debugger session here teaches more than any document, including this one.

*Files:* vllm/v1/engine/core.py · vllm/v1/engine/core_client.py (InprocClient)

### 2 · Watch a long prompt get processed in slices  *(warm-up)*

Shrink the per-round token budget so one prompt needs several rounds, and log what the scheduler hands out:

```bash
# add one line at the end of Scheduler.schedule() (scheduler.py, just before return):
logger.info("round: %s", {r: n for r, n in scheduler_output.num_scheduled_tokens.items()})

VLLM_ENABLE_V1_MULTIPROCESSING=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-0.6B', max_num_batched_tokens=64, max_model_len=2048,
          enable_chunked_prefill=True, gpu_memory_utilization=0.5)
llm.generate(['word ' * 400], SamplingParams(max_tokens=4))"
```

**Expect:** The same request showing up round after round, getting 64 tokens each time, until its ~400-token prompt is used up. That's "chunked prefill": not a mode, just a budget that ran out. Undo the edit when you're done.

*Files:* vllm/v1/core/sched/scheduler.py:477 (schedule) · :1383 (_update_after_schedule)

### 3 · Prove the cache reuse with numbers  *(warm-up)*

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-0.6B', max_model_len=4096, gpu_memory_utilization=0.5,
          disable_log_stats=False)
prefix = 'You are a careful assistant. ' * 40
for q in ['What is 2+2?', 'Name a prime.']:
    llm.generate([prefix + q], SamplingParams(max_tokens=8))
for m in llm.get_metrics():
    if 'prefix' in m.name: print(m.name, getattr(m, 'value', None))"
```

**Expect:** Almost no cache hits on the first call, then hits ≈ queries on the second — the shared opening was reused and only the differing tail was computed. Run again with `enable_prefix_caching=False` and watch the counters go silent.

*Files:* vllm/v1/core/kv_cache_utils.py:618 (the fingerprint math) · kv_cache_manager.py:232 (the lookup)

### 4 · Race the attention implementations on your 5090  *(medium)*

Same model, three implementations, real numbers from your own card:

```bash
for B in FLASH_ATTN FLASHINFER TRITON_ATTN; do
  echo "=== $B ==="
  vllm bench latency --model Qwen/Qwen3-0.6B --attention-backend $B \
    --input-len 512 --output-len 128 --num-iters 5 --gpu-memory-utilization 0.5
done
```

**Expect:** FLASH_ATTN and FLASHINFER close together, TRITON_ATTN behind. Then read the code that made FLASH_ATTN the default for your card — and remember it's running FlashAttention 2 while server chips get 3 and 4. That gap is contribution territory.

*Files:* vllm/platforms/cuda.py:80 (the ranked lists) · vllm/v1/attention/backend.py:353 (the checks)

### 5 · Open the compiled graph and find the cuts  *(medium)*

```bash
python -c "
from vllm import LLM
LLM(model='Qwen/Qwen3-0.6B', max_model_len=2048, gpu_memory_utilization=0.5)"
ls ~/.cache/vllm/torch_compile_cache/*/rank_0_0/backbone/
${EDITOR:-less} ~/.cache/vllm/torch_compile_cache/*/rank_0_0/backbone/computation_graph.py
```

**Expect:** The model's traced graph after cutting: compiled pieces alternating with `unified_attention*` calls — the cuts from §8, in the flesh, roughly one per layer. Start once with `--optimization-level 0` for contrast: no cache directory appears at all.

*Files:* vllm/compilation/backends.py:1020 · :553 (split_graph)

### 6 · Force a request to get kicked out  *(medium)*

Give the cache pool almost nothing, then ask for more than fits:

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-0.6B', max_model_len=2048,
          gpu_memory_utilization=0.5, kv_cache_memory_bytes=64*2**20)
outs = llm.generate(['Write a long story. '] * 16, SamplingParams(max_tokens=1500))
for m in llm.get_metrics():
    if 'preempt' in m.name: print(m.name, getattr(m, 'value', None))"
```

**Expect:** A preemption count above zero and a warning in the log. Then read `_preempt_request` and match it to §4: memory freed, progress reset to zero, request back at the front of the line — and the cache quietly absorbing most of the redo on its way back.

*Files:* vllm/v1/core/sched/scheduler.py:1340 (_preempt_request)

### 7 · Plug in your own model class  *(deeper)*

The same hook a real "add a new model" PR uses — without the PR:

```bash
python -c "
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.llama import LlamaForCausalLM

class MyLlama(LlamaForCausalLM):   # pretend new architecture
    pass

ModelRegistry.register_model('MyLlamaForCausalLM', MyLlama)
llm = LLM(model='Qwen/Qwen3-0.6B', max_model_len=1024, gpu_memory_utilization=0.5,
          hf_overrides={'architectures': ['MyLlamaForCausalLM']})
print(llm.generate(['hi'])[0].outputs[0].text)"
```

**Expect:** The model loads through *your* class (Qwen3 is close enough to Llama for a smoke test). Then skim `docs/contributing/model/` — the checklist for adding a real architecture, one of the most common first contributions.

*Files:* vllm/model_executor/models/registry.py · vllm/model_executor/model_loader/utils.py:38

### 8 · Scout your first real contribution  *(deeper)*

```bash
gh issue list --repo vllm-project/vllm --label "good first issue" --state open --limit 20
gh issue list --repo vllm-project/vllm --search "sm120 OR blackwell OR 5090 in:title,body" --state open --limit 20
# before touching anything — the required check that nobody's already on it:
gh pr list --repo vllm-project/vllm --state open --search "<issue number> in:body"
```

**Expect:** A shortlist. Before writing any code, run the tests for that area locally (`uv pip install -r requirements/test/cuda.in`, then `.venv/bin/python -m pytest tests/… -v`). The project's rules for contributors are strict — next section.

*Files:* AGENTS.md · docs/contributing/

## §11 House rules

The repo ships an `AGENTS.md` that is short, strict, and enforced. The gist:

- **Check for duplicates first, always.** Search open PRs for the issue number and the topic. If someone's already fixing it, don't open a second PR.
- **No trivial PRs.** A lone typo fix or style tweak will be closed; small cleanups ride along with real work.
- **Say when AI helped, and own every line.** Agent-written PRs where no human understands the change are banned. You review it all, you run the tests, you can defend it.
- **The PR description must show your work**: why it's not a duplicate, which tests you ran and their results, and quality-benchmark results for anything that changes model output.
- **Practical bits**: always `uv` and the repo's venv, never bare pip. Run `pre-commit run` before pushing (the hooks are already installed in your checkout). Lines max 88 characters. Google-style docstrings.

## §12 Reading list, ranked

The twelve files that teach the most per line, in the order to read them:

| # | File | Why |
|---|---|---|
| 1 | `vllm/v1/engine/core.py` | The main loop. Once you know this file, you know the shape of everything. |
| 2 | `vllm/v1/core/sched/scheduler.py` | Every decision: budgets, admission, kicking requests out. |
| 3 | `vllm/v1/core/kv_cache_utils.py` | The block fingerprints and the free list — §5 in code form. |
| 4 | `vllm/v1/worker/gpu_model_runner.py` | Huge; just follow one round: _update_states → _prepare_inputs → execute_model. |
| 5 | `vllm/v1/sample/sampler.py` | Short and readable; the step order is the contract. |
| 6 | `vllm/model_executor/layers/attention/attention.py` | How a layer finds its implementation, its memory, and its per-round data. |
| 7 | `vllm/forward_context.py` | The shared per-round object that makes the compilation trick possible. |
| 8 | `vllm/config/vllm.py` | Where all the configs meet and get checked against each other. |
| 9 | `vllm/compilation/backends.py` | Trace, cut, compile, cache — §8 in code form. |
| 10 | `vllm/platforms/cuda.py` | Every "on this GPU, do it this way" decision, in one file. |
| 11 | `vllm/v1/engine/async_llm.py` | The front-end half: streaming, cancellation, back-pressure. |
| 12 | `vllm/v1/spec_decode/llm_base_proposer.py` | All of speculative decoding's draft models share this one base class. |

Built from a seven-agent read of vllm-project/vllm @ 4b7cb94 · 2026-08-20 · paths checked against your checkout — line numbers drift as main moves.

