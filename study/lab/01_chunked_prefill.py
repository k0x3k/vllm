"""Watch one real prompt get chewed through in budget-sized slices.

Atlas §4: there is no "prefill mode". Every request just has a gap between
tokens computed and tokens needed, and each round shares out one budget.
"""

from _lab import fake_model_output, make_request, tokenize

BUDGET = 32  # max_num_batched_tokens: deliberately tiny so slicing is visible

text = "The quick brown fox jumps over the lazy dog. " * 12
token_ids = tokenize(text)
print(f"prompt: {len(token_ids)} tokens, per-round budget: {BUDGET}\n")

from _lab import create_scheduler  # noqa: E402

scheduler = create_scheduler(
    max_num_batched_tokens=BUDGET, num_blocks=256, block_size=16, max_model_len=2048
)
req = make_request("story", token_ids, max_tokens=4)
scheduler.add_request(req)

for round_no in range(1, 30):
    before = req.num_computed_tokens
    out = scheduler.schedule()
    if not out.num_scheduled_tokens:
        break
    handed_out = out.num_scheduled_tokens[req.request_id]
    done = req.num_computed_tokens
    phase = "prefill" if before < len(token_ids) else "decode"
    blocks = len(scheduler.kv_cache_manager.get_block_ids(req.request_id)[0])
    print(
        f"round {round_no:>2}  {phase:<7} budget used {handed_out:>2}/{BUDGET}  "
        f"prompt {min(done, len(token_ids))}/{len(token_ids)}  blocks {blocks}"
    )
    scheduler.update_from_output(out, fake_model_output(scheduler))
    if req.is_finished():
        print(f"\nfinished: {req.status}")
        break
