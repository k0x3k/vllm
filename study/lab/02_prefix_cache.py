"""Prove the prefix cache with numbers, not vibes.

Atlas §5: every full block gets a fingerprint chained from the previous block's
fingerprint, so one match proves the whole prefix matched. The lookup itself is
KVCacheManager.get_computed_blocks.
"""

from _lab import create_scheduler, fake_model_output, make_request, tokenize

BLOCK = 16
SHARED = "You are a careful assistant. Answer precisely and briefly. " * 8

scheduler = create_scheduler(
    enable_prefix_caching=True, num_blocks=256, block_size=BLOCK, max_model_len=2048
)
manager = scheduler.kv_cache_manager

shared_len = len(tokenize(SHARED))
print(f"shared prefix: {shared_len} tokens = {shared_len // BLOCK} full blocks of {BLOCK}\n")


def run_to_completion(req_id: str, text: str) -> None:
    req = make_request(req_id, tokenize(text), max_tokens=2, block_size=BLOCK)
    blocks, cached_tokens, _ = manager.get_computed_blocks(req)
    print(f"{req_id:<10} lookup before admission: {cached_tokens:>3} tokens already cached")
    scheduler.add_request(req)
    while not req.is_finished():
        out = scheduler.schedule()
        if not out.num_scheduled_tokens:
            break
        scheduler.update_from_output(out, fake_model_output(scheduler))


run_to_completion("first", SHARED + "What is 2+2?")
run_to_completion("second", SHARED + "Name a prime number.")
run_to_completion("unrelated", "Completely different opening text entirely.")

print(f"\nblock pool usage: {manager.block_pool.get_usage():.1%}")
