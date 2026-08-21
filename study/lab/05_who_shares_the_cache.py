"""Who is allowed to reuse whose KV cache?

A block's fingerprint is hash(parent_fingerprint, token_ids, extra_keys).
No request id, no session id, no user id goes in — only what the model would
actually compute. So identical text always shares, whoever sent it.

extra_keys is the deliberate escape hatch: LoRA id, multimodal item hashes,
and cache_salt. Salt is applied only to block 0, which is enough, because the
chain carries it forward to every later block.
"""

from _lab import create_scheduler, fake_model_output, make_request, tokenize

BLOCK = 16
scheduler = create_scheduler(
    enable_prefix_caching=True, num_blocks=64, block_size=BLOCK, max_model_len=512
)
PROMPT = "You are a helpful assistant. Answer in one short sentence. What is the capital of France?"
ids = tokenize(PROMPT)
print(f"identical prompt, {len(ids)} tokens\n")


def run(label, salt=None):
    req = make_request(label, ids, max_tokens=2, block_size=BLOCK, cache_salt=salt)
    _, hit, _ = scheduler.kv_cache_manager.get_computed_blocks(req)
    print(f"{label:<28} salt={str(salt):<10} cache hit: {hit:>2} tokens")
    scheduler.add_request(req)
    while not req.is_finished():
        out = scheduler.schedule()
        if not out.num_scheduled_tokens:
            break
        scheduler.update_from_output(out, fake_model_output(scheduler))


run("alice (first ever)")
run("bob (different user)")
run("carol (same text, salted)", salt="tenant-carol")
run("dave (same salt as carol)", salt="tenant-carol")
run("erin (different salt)", salt="tenant-erin")
