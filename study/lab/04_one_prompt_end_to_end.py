"""Follow one prompt from text to freed blocks, then a second chat turn.

Answers two questions: where exactly do blocks get allocated and freed, and
what a multi-turn chat looks like to vLLM (spoiler: it has no idea what a
chat is).
"""

from _lab import create_scheduler, fake_model_output, make_request, tokenize

BLOCK = 16
scheduler = create_scheduler(
    enable_prefix_caching=True, num_blocks=64, block_size=BLOCK, max_model_len=512
)
pool = scheduler.kv_cache_manager.block_pool


def free_list():
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]


def cached_hashes():
    return len(pool.cached_block_hash_to_block)


SYSTEM = "You are a helpful assistant. Answer in one short sentence. "
TURN1 = SYSTEM + "The capital of France is"


def run(label, text):
    ids = tokenize(text)
    req = make_request(label, ids, max_tokens=3, block_size=BLOCK)

    _, hit, _ = scheduler.kv_cache_manager.get_computed_blocks(req)
    print(f"\n--- {label}: {len(ids)} tokens ---")
    print(f"  prefix-cache lookup      : {hit} tokens already computed")

    scheduler.add_request(req)
    while not req.is_finished():
        out = scheduler.schedule()
        if not out.num_scheduled_tokens:
            break
        blocks = scheduler.kv_cache_manager.get_block_ids(req.request_id)[0]
        print(f"  scheduled {out.num_scheduled_tokens[label]:>3} tok -> blocks {blocks}")
        scheduler.update_from_output(out, fake_model_output(scheduler))

    print(f"  finished ({req.status.name}) -> _free_blocks() called")
    print(f"  free list now            : {free_list()[:6]}...")
    print(f"  fingerprints still alive : {cached_hashes()} blocks worth of reusable KV")


print(f"pool: {len(free_list())} free blocks, {cached_hashes()} fingerprints")
run("turn1", TURN1)
run("turn2", TURN1 + " Paris. Now name a French river.")
