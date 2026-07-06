# `amass/tier/` — the DRAM KV-offload tier (AMASS-mem-v / mem-kv)

The memory play: the V half (mem-v) or K+V (mem-kv) of the KV cache lives in a
pinned, device-mapped **host DRAM** pool; only a bounded **hot buffer** +
**staging pool** stay in VRAM. This is the clean rewrite of the prototype's
`vllm/kernels/dram_tier.py` (1762 lines of debug cruft), restructured so the two
confirmed mem bugs are fixed **by construction** via four written invariants.

STATUS 2026-07: the tier is **functionally complete and gated at the kernel /
module level** — write path, flush lifecycle, exact-LRU residency, miss fetch,
and the DYNAMIC-budget tiered decode (`decode_mem.py`) run end-to-end and are
BITWISE-equal to the resident decode through multi-step chunked-prefill +
decode lifecycles (mns 1/2/4, forced eviction, CUDA-graph replay; see
`tests/test_mem_dynamic.py`). The remaining follow-up is the vLLM SERVING
integration (below), not tier machinery.

## Module layout

| module | owns | status |
|---|---|---|
| `pool.py` | `MappedHostVPool` — pinned host K/V pools + UVA device pointer | **real** (UVA, sizing, cap) |
| `residency.py` | `Residency` — hot buffer, exact-LRU maps, graph-safe plan + flat-list miss fetch (I2/I3) | **real** (Triton, graph-safe, variable-count native) |
| `staging.py` | `Staging` — staging pools + per-step flush/alloc lifecycle (I1/I4) | **real** (Triton, graph-safe; free-after-flush for ALL requests keeps the O(chunk) bound) |
| `tier.py` | `Tier` facade + `TierVSource` (the `V_SRC==1` seam) | **real** (begin_step skips mutation on capture batches) |
| `decode_mem.py` | the DYNAMIC-budget mem tiered decode: variable per-unit fetch + ptr-select split-K decode | **real** (bitwise vs resident; graph-capturable) |

## The dynamic-budget decode (`decode_mem.py`)

`mem_dynamic_decode(q, kv_cache, block_table, seq_lens, st, out, tier, lidx)`
consumes the Stage-A interface `st.page_table` (R, n_kv, MP; ascending, -1
padded, sink+window+tail forced) + `st.page_cnt` (R, n_kv; **VARIABLE** per
(layer, kv-head) unit under the dynamic budget), and per layer:

1. `tier.step` — graph-safe residency gather: miss classify + exact-LRU
   victims (`gather_plan`), then the **flat-list fetch** (`gather_fetch`): the
   victims kernel compacts every installed miss into a `(kv, blk, slot)` list
   and a small strided grid (1024 programs) copies it — launch cost decoupled
   from `n_req*max_pages` and page-level load-balanced across any b_u skew.
2. the **ptr-select** tiered split-K decode + merge: the hot|staged|pinned
   3-way select is computed once per page on the ADDRESS (all sources
   addressed as int16 bits), one load per K/V tile. Bitwise-equal to the
   shared `V_SRC==1` seam kernel and to the resident decode.

Measured (H200, 4x16K reqs, budget 0.10, 98% steady hot-hit, CUDA-graph
replay): plan 7.1us + fetch 2.3us + decode 31.6us per layer-step vs 29.6us
resident decode. Fetch transport = UVA kernel at 45-50 GiB/s = **84-92% of the
contiguous-PCIe roofline (53.7 GiB/s)**; copy-engine DMA on scattered top-b
pages is launch-bound (~1.6 GiB/s over ~1-page runs) and stays reserved for
the `fetch="dma"` escape hatch.

## The invariant contract (`AMASS_DESIGN.md` §8.3)

Each invariant is **structurally** enforced — a single owner of the transition
and/or a fixed call order — not merely hoped, with a cheap debug assertion for
the content-independent half.

- **I1 (no lost V)** — every COMPLETE written page has V retrievable: STAGED
  (`vbo[blk]>=0`, in `v_pool`) OR FLUSHED (`pool_valid=1` all layers/heads, in
  the pinned pool). *Enforced:* `Staging._valloc` guarantees a v-block for every
  written page and **raises on free-stack underflow** (never silently `vbo=-1`);
  a v-block is freed **only after** its flush copy is enqueued. *Bug #1* was
  exhaustion → `vbo=-1` AND `pool_valid=0` → zeros. *Check:* `assert_no_lost_v`,
  `assert_capacity`.
- **I2 (hot == residency)** — `hot[l,kv,slot]` holds `slot2page[l,kv,slot]`'s V.
  *Enforced:* the gather is the **single owner** of the `(hot[slot], page2slot,
  slot2page)` transition and reads from the correct source (pinned if
  `pool_valid` else staged via `vbo`). *Bug #2* was a gather filling hot with the
  wrong page's V for pages that stayed STAGED into decode. *Check:*
  `assert_maps_inverse` (the addressing half; content half is single-owner
  by construction).
- **I3 (no aliasing)** — free-stack is a true set: assigned(`vbo>=0`) + free
  (`v_free_stack[:top]`) partition `[0,NV)` with no duplicates; `page2slot` /
  `slot2page` are mutual inverses. *Enforced:* single owner of push/pop (CAS in
  the kernel). *Check:* `assert_free_partition`, `assert_maps_inverse`.
- **I4 (flush-before-reuse ordering)** — within `begin_step`, `flush_copy` reads
  `v_pool[vb]` **before** `valloc` reassigns `vb` and **before** this step's
  scatter overwrites it. *Enforced:* the fixed call order inside the single
  `Staging.begin_step` method — the order **is** the mechanism.

Set `AMASS_TIER_ASSERT=1` to run the (host-syncing) asserts inside every
`begin_step`.

## Sizing discipline (the flagged 17GiB boundary)

- **HOST pinned pool** (`pool.py`) is keyed by physical block, sized by
  `num_gpu_blocks` — **by design**: it is the DRAM offload target and must hold
  every flushed page's V (I1). Host DRAM is abundant/cheap; capped by
  `max_pool_gb` (raises, never truncates).
- **VRAM** structures (`hot` buffer S, staging `v_pool` NV) are bounded to the
  **working set** — `hot_slots`, `2*max_pages + 4*max_reqs` — **never**
  `num_gpu_blocks`. The 17GiB bug was VRAM per-block state sized by
  `num_gpu_blocks` (fast path's R8State). The only NB-sized VRAM here is tiny
  int8/int32 residency bookkeeping (~1-4 B/block).

## The `TierVSource` seam (§8.1)

`attention/decode.py` stays tier-blind: it loads V through a `VSource` with a
compile-time `SRC_ID`. `ResidentVSource` (fast) is `V_SRC==0`; `TierVSource`
(mem-v) is `V_SRC==1`. `TierVSource.args()` satisfies the shared wrapper's
4-tuple contract; `TierVSource.union()` exposes the full 3-way pointer union the
follow-up decode kernel consumes (hot | staged v_pool | pinned pool).

## Done vs FOLLOW-UP

**Done (real, gated by `tests/test_mem_dynamic.py`, all Triton, graph-safe):**
every persistent buffer + sizing; `pool.py` UVA mapping; the full `Staging`
lifecycle (`_vflush_clear` / `_flush_collect` / `_flush_copy` / `_valloc` /
`_vslot` — bit-exact int16 flush copies; **free-after-flush for ALL requests**,
so chunked prefill of any length stays inside the O(chunk) staging bound);
`Tier.write_kv` scatter (`_scatter_kv_kernel`, K+V for mem-kv);
`Residency.invalidate_written` (skipped on cudagraph-capture dummy batches —
a dummy block table must never drop live residency/validity);
`Residency.gather` = `gather_plan` (miss diff + exact-LRU victims + flat-list
compaction) + `gather_fetch` (strided flat fetch, variable-count native);
the `V_SRC==1`/`K_SRC==1` 3-way load in `attention/decode.py` AND the faster
ptr-select tiered decode in `decode_mem.py`; copy-engine DMA
(`gather_fetch_dma`, correctness-gated; launch-bound on scattered pages —
kernel transport is the default).

Gates: bitwise mem==resident through chunked-prefill + multi-step decode
(mem-kv and mem-v), matched-mns {1,2,4}, forced eviction (pinned-fallback
decode), DMA transport, CUDA-graph capture/replay, real Llama-3.1-8B 28K K +
real decode queries, I1-I4 invariants every step, dynamic>=static retained
mass.

**FOLLOW-UP (vLLM serving integration; the tier machinery is ready):**
1. **Impl wiring** — `backend/attn.py::_forward_mem`: call the Stage-A
   selection, `tier.write_kv` on every step (prefill + decode), then
   `tier.step` + the tiered decode (`amass.tier.mem_dynamic_decode`) on pure
   decode. Needs mem-variant Stage-A state in the builder (the r8/quad state
   is currently fast-only).
2. **K-only spec patch** — `backend/register._patch_kv_cache_spec_k_only`
   (`head_size_v=0`), applied together with (1) so attention never loses its
   V half without a tier replacement (this is what realizes the VRAM saving).
3. Flip `_runtime._MEM_STAGEB_WIRED = True` and validate with
   `scratch_dram_repro/sys_equality.py` at matched mns=1 and mns>=3 (the
   module-level matched-mns gate in `tests/test_mem_dynamic.py` is the
   kernel-layer half of that gate and is green).
