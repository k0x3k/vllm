"""Station 1: where your string stops being a string.

Runs the real front-end code path — the same objects LLM.generate() builds,
minus the engine. Put breakpoint() anywhere below and step in.

Call chain this reproduces:
  LLM.generate                       entrypoints/llm.py:418
    _run_completion                  entrypoints/offline_utils.py:326
      _add_completion_requests       entrypoints/offline_utils.py:270
        _preprocess_cmpl_one -> _preprocess_cmpl
          parse_model_prompt         renderers/inputs/preprocess.py:235
          renderer.render_cmpl       renderers/base.py:980
        _render_and_add_requests -> _add_request -> llm_engine.add_request
"""

import os

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from vllm.engine.arg_utils import EngineArgs
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import parse_model_prompt

TEXT = "The capital of France is"

cfg = EngineArgs(model="facebook/opt-125m", max_model_len=256).create_engine_config()
renderer = renderer_from_config(cfg)
print(f"renderer chosen for this model: {type(renderer).__name__}\n")

# 1. Shape normalisation. A str, a {"prompt": ...} dict, a raw token list and a
#    multimodal dict all leave here in the same dict shape.
parsed = parse_model_prompt(cfg.model_config, TEXT)
print(f"1. parse_model_prompt   {TEXT!r}\n                     -> {parsed}\n")

# 2. render_cmpl: stamp arrival_time, render -> tokenize -> package.
engine_input = renderer.render_cmpl([parsed])[0]
print("2. render_cmpl -> EngineInput (a plain dict):")
for k, v in engine_input.items():
    print(f"     {k:<18} {v!r}"[:100])

ids = engine_input["prompt_token_ids"]
tok = renderer.tokenizer
print(f"\n3. {len(ids)} token ids, one at a time:")
for i in ids:
    print(f"     {i:>6}  {tok.decode([i])!r}")
print("\n   note id 2 = BOS, prepended by the tokenizer. It was never in your string,")
print("   it costs a KV cache slot, and it counts against max_model_len.")
