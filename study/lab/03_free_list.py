"""There is no eviction algorithm — LRU is an emergent property of list order.

Atlas §5. BlockPool.free_blocks pushes fingerprinted blocks to the BACK of the
free list and unfingerprinted ones to the FRONT; allocation always takes from
the front. That ordering is the entire policy.

Try to break it: swap the two lists in BlockPool.free_blocks (block_pool.py:719)
and predict what this script prints before you rerun it.
"""

from _lab import create_scheduler, fake_model_output, make_request, tokenize

BLOCK = 16

scheduler = create_scheduler(
    enable_prefix_caching=True, num_blocks=32, block_size=BLOCK, max_model_len=512
)
pool = scheduler.kv_cache_manager.block_pool


def free_list() -> list[int]:
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]


def fingerprinted() -> list[int]:
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks() if b.block_hash]


print(f"fresh pool, free list head: {free_list()[:8]} ... ({len(free_list())} free)\n")

req = make_request("a", tokenize("Cache this prompt. " * 10), max_tokens=2, block_size=BLOCK)
scheduler.add_request(req)
while not req.is_finished():
    out = scheduler.schedule()
    if not out.num_scheduled_tokens:
        break
    scheduler.update_from_output(out, fake_model_output(scheduler))

print(f"after one request finished and was freed:")
print(f"  free list head (next to be handed out): {free_list()[:8]}")
print(f"  fingerprinted blocks still holding cache: {fingerprinted()}")
print()
print("The fingerprinted blocks sit at the BACK, so they are the last to be")
print("reused — that is the LRU eviction, with no eviction code anywhere.")
