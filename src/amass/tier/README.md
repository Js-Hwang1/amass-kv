# `amass/tier/` — the DRAM V-offload tier (AMASS-mem-v)

The memory play: the V half of the KV cache lives in a pinned, device-mapped
**host DRAM** pool; only a bounded **hot buffer** + **staging pool** stay in
VRAM. K stays resident. This is the clean rewrite of the prototype's
`vllm/kernels/dram_tier.py` (1762 lines of debug cruft), restructured so the two
confirmed mem bugs are fixed **by construction** via four written invariants.

This directory is a **scaffold**: real, coherent interfaces + the
invariant-enforcing skeleton that compiles and imports cleanly. The V-byte-moving
kernels are simple/torch references or documented follow-up hooks.

## Module layout

| module | owns | status |
|---|---|---|
| `pool.py` | `MappedHostVPool` — pinned host V pool + UVA device pointer | **real** (UVA, sizing, cap) |
| `residency.py` | `Residency` — hot buffer, exact-LRU maps, gather (I2/I3) | buffers + asserts real; `gather` = follow-up |
| `staging.py` | `Staging` — v_pool + per-step flush/alloc lifecycle (I1/I4) | buffers + `begin_step` ordering + asserts real; sub-step bodies = torch refs (TODO kernelize) |
| `tier.py` | `Tier` facade + `TierVSource` (the `V_SRC==1` seam) | facade + begin_step + invariant checks real; write/step/vsource-union = follow-up hooks |

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

## Scaffolded vs stubbed vs FOLLOW-UP

**Scaffolded (real):** all persistent buffers + sizing; `pool.py` UVA mapping;
`Tier`/`Staging`/`Residency` construction; `begin_step` orchestration + the I4
call order; every invariant assertion; the `TierVSource`/`union()` seam; the
backend wiring (builder allocates the tier + drives `begin_step` outside the
graph; impl builds the seam and delegates to stock FA until the tiered load
lands; readiness flag `_runtime.mem_stageb_wired()`).

**Stubbed (torch reference / no-op, TODO kernelize):** `Staging` sub-steps
(`_vflush_clear` / `_flush_collect` / `_flush_copy` / `_valloc` / `_vslot`) are
correct but host-syncing; `_flush_copy`'s actual byte copy is a no-op that only
advances validity (state machine testable; bytes not yet in the pinned pool);
`Residency.invalidate_written` is a minimal torch reference.

**FOLLOW-UP (to make mem-v decode end-to-end):**
1. **`V_SRC==1` decode load** — flesh the documented `else` arm in
   `attention/decode.py` **and** `attention/decode_cuda.py` into the 3-way read
   (hot via `page2slot`+complete, staged via `vbo`, pinned via `pool_off`),
   keyed by `pool_valid`+`vbo`, taking the extra pointers from
   `TierVSource.union()`. Reference: `_p2_sparse_decode_split_kernel` in
   `vllm/kernels/dram_tier.py`. *(Owned by the decode/selection agent — do not
   edit those two files from the tier side.)*
2. **`Tier.write_kv`** — in-graph scatter K→engine (K-only) / V→`v_pool` via
   `v_slot_mapping`, replacing `reshape_and_cache_flash`. Port
   `_p2_scatter_kv_kernel`.
3. **`Residency.gather` + `Tier.step`** — graph-safe per-layer gather into `hot`.
   Port `_vt_miss_diff_kernel` + `_select_victims_kernel` + `_p2_gather_kernel`.
4. **Graph-safe port** of the `Staging` sub-step bodies + the real `_flush_copy`
   bit-exact host copy (`_p2_*` kernels).
5. **K-only spec patch** — `backend/register._patch_kv_cache_spec_k_only`
   (`head_size_v=0`), applied together with (1) so attention never loses its V
   half without a tier replacement.
6. Flip `_runtime._MEM_STAGEB_WIRED = True` and validate with
   `scratch_dram_repro/sys_equality.py` at matched mns=1 and mns>=3.
