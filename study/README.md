# Study layer

Personal learning layer for this fork of `vllm-project/vllm`. Lives on the `study`
branch only — `main` stays a clean mirror of upstream so PR branches cut from it
carry none of this.

## Where things are

| What | Where |
|---|---|
| Field guide to the codebase | [`study/vllm-atlas.md`](vllm-atlas.md) — read this first |
| Topic notes | [`study/notes/`](notes/) |
| Session transcripts + Claude memory | private repo `k0x3k/vllm-lab`, symlinked into `~/.claude/projects/` |
| Machine bootstrap | [`study/lab-setup.sh`](lab-setup.sh) — run once per machine/rebuild |
| House rules for contributing | [`AGENTS.md`](../AGENTS.md) — strict, read before any PR |

## The shape of vLLM, in six lines

A running server is **three programs**, and knowing which one you are in is the
most useful habit while reading:

- **Front end** — HTTP/`LLM` API, text→tokens, tokens→text. `vllm/entrypoints/`, `vllm/v1/engine/async_llm.py`
- **Engine** — one loop, all the decisions, never touches the GPU. `vllm/v1/engine/core.py`, `vllm/v1/core/sched/scheduler.py`
- **Workers** — one per GPU, the only GPU code. `vllm/v1/worker/gpu_model_runner.py`

`VLLM_ENABLE_V1_MULTIPROCESSING=0` folds all three into one debuggable process.

Start with the top of the ranked reading list in the atlas: `core.py` (the loop),
`scheduler.py` (the decisions), `gpu_model_runner.py` (the execution). Everything
else hangs off those three.

## Session persistence

`study/lab-setup.sh` symlinks `~/.claude/projects/<slug>` into a clone of the
private `vllm-lab` repo and installs a `SessionEnd` hook that commits and pushes.

In a Codespace the built-in `GITHUB_TOKEN` is scoped to this repo alone and
cannot push to `vllm-lab`. Create a fine-grained PAT with **Contents: read and
write** on `k0x3k/vllm-lab` and expose it as `VLLM_LAB_TOKEN` (a Codespaces
secret survives rebuilds); both scripts pick it up automatically.

## Working on this fork

```bash
git fetch upstream && git checkout main && git merge --ff-only upstream/main   # keep main clean
git checkout study && git rebase main                                          # carry the study layer forward
git checkout -b fix/whatever main                                              # PR branches come off main
```

## Notes for Claude

- Line numbers in the atlas drift as `main` moves — verify before quoting them.
- Never use system `python3` or bare `pip`; everything goes through `uv` and `.venv/bin/python`.
- This is a public fork. Nothing secret goes in `study/`; transcripts go to the private lab repo.
