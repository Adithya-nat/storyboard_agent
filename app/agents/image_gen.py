"""image_gen agent: Shotlist shots -> one CLEAN frame image per shot.

Three phases so we NEVER ship a gap OR an ugly placeholder card to the user:

    Phase A  attempt (bounded pool)   Skip clean on-disk frames (resume/idempotency).
             Generate the rest conditioned on style_ref. Collect CLEAN bytes / failures.
    Phase B  fill gaps                Each failure -> reuse the nearest successful
             neighbor's CLEAN bytes (no gap, no jarring card). All failed, no
             neighbor -> a styled placeholder card (base case only).
    Phase C  save                     Write CLEAN frame_NN.png. Failed/filled keep a
             .degraded marker so resume re-attempts real generation, and report
             status=degraded (honest in GET).

Frames are saved CLEAN (no burned caption): the assembler burns/overlays the
caption later, so a borrowed neighbor never carries the wrong caption, and Wave 2
(caption as ffmpeg overlay) needs no frame changes.

Concurrency: an asyncio.Semaphore bounds how many image calls run at once; the
shared token bucket inside the adapter bounds the per-minute rate (so re-rolls +
concurrency provably stay under the model's RPM). Every frame is conditioned on the
designer's single style_ref.png — the load-bearing consistency lever.
"""

from __future__ import annotations

import asyncio

from app.adapters.errors import ProviderError
from app.core.images import render_card
from app.core.logging import get_logger
from app.graph.state import Deps, FrameMeta, StoryboardState
from app.schemas import Shot, Shotlist

log = get_logger(__name__)


async def image_gen(state: StoryboardState, deps: Deps) -> StoryboardState:
    store = deps.store
    reporter = deps.reporter  # None in pure unit tests -> all calls below are skipped
    shotlist = Shotlist.model_validate(store.load_json(state["shotlist_path"]))
    shots = sorted(shotlist.shots, key=lambda s: s.order)
    existing_orders = {s.order for s in shots}
    total = len(shots)

    # Load the one style reference once (None => designer fell back to text-only).
    style_ref_path = state.get("style_ref_path")
    ref_image = store.load_bytes(style_ref_path) if style_ref_path else None

    sem = asyncio.Semaphore(max(1, deps.settings.image_max_concurrency))

    if reporter is not None:
        reporter.log(
            f"generating {total} frame(s) "
            f"(≤{deps.settings.image_max_concurrency} concurrent)"
        )
        await reporter.progress(0, total)

    # --- Phase A: attempt every missing/degraded shot (bounded pool) -------- #
    clean_bytes: dict[int, bytes] = {}   # generated THIS run (un-captioned)
    reused_clean: set[int] = set()       # already clean on disk -> skipped, kept
    failures: dict[int, str] = {}
    done = 0  # frames resolved (generated / reused / failed) -> drives progress

    async def attempt(shot: Shot) -> None:
        nonlocal done
        frame_name = f"frame_{shot.order:02d}.png"
        marker_name = f"frame_{shot.order:02d}.degraded"
        # Idempotency (resume): a clean frame already on disk is kept, not re-billed.
        if store.exists(frame_name) and not store.exists(marker_name):
            reused_clean.add(shot.order)
            if reporter is not None:
                reporter.log(f"frame {shot.order}/{total}: reused clean frame (resume)")
        else:
            async with sem:
                try:
                    raw = await asyncio.to_thread(
                        deps.image.generate_image,
                        shot.image_prompt,
                        ref_image=ref_image,
                        seed=deps.settings.image_seed,
                    )
                    clean_bytes[shot.order] = raw
                    if reporter is not None:
                        reporter.log(f"frame {shot.order}/{total}: generated")
                except ProviderError as exc:
                    # The bridge captures this log.warning into job_logs (image_gen
                    # stage) via the contextvar — no explicit reporter.log needed.
                    log.warning("frame %s generation failed after retries: %s", shot.order, exc)
                    failures[shot.order] = str(exc)
        done += 1  # single-threaded event loop: safe (no await between read+write)
        if reporter is not None:
            await reporter.progress(done, total)

    await asyncio.gather(*(attempt(s) for s in shots))

    # QA toggle: force one degraded frame so the completed_with_warnings UI is
    # testable (mock mode otherwise only ever yields clean frames). Treat the first
    # shot as a failure -> Phase B neighbour-fills it -> status=degraded.
    if deps.settings.mock_force_degraded_frame and shots:
        victim = shots[0].order
        clean_bytes.pop(victim, None)
        reused_clean.discard(victim)
        failures.setdefault(victim, "forced degraded frame (QA toggle)")

    # --- Phase B helpers: nearest CLEAN neighbor (memory or clean on disk) -- #
    def clean_source(order: int) -> bytes | None:
        if order in clean_bytes:
            return clean_bytes[order]
        if order in reused_clean:
            return store.load_bytes(store.abspath(f"frame_{order:02d}.png"))
        return None

    def nearest_clean(order: int) -> bytes | None:
        available = [o for o in existing_orders if clean_source(o) is not None]
        if not available:
            return None
        best = min(available, key=lambda o: (abs(o - order), o))
        return clean_source(best)

    # --- Phase C: save clean frames; fill gaps seamlessly ------------------- #
    frames: list[FrameMeta] = []
    warnings: list[str] = []
    for shot in shots:
        order = shot.order
        frame_name = f"frame_{order:02d}.png"
        marker_name = f"frame_{order:02d}.degraded"

        if order in reused_clean:
            frames.append(
                FrameMeta(order=order, path=store.abspath(frame_name),
                          caption=shot.narration, status="ok")
            )
            continue

        if order in clean_bytes:
            path = store.save_bytes(frame_name, clean_bytes[order])
            store.remove(marker_name)  # clear any stale marker from a prior degraded run
            frames.append(FrameMeta(order=order, path=path, caption=shot.narration, status="ok"))
            continue

        # Failed -> seamless neighbor-fill (no gap, no ugly card). Only the
        # all-frames-failed base case (no clean neighbor anywhere) shows a placeholder.
        fill = nearest_clean(order)
        if fill is not None:
            data, how = fill, "reused a neighbouring frame"
        else:
            data, how = render_card(shot.image_prompt, shot.narration, degraded=True), "used a placeholder"
        path = store.save_bytes(frame_name, data)
        store.write_marker(marker_name)  # survive restart -> re-try real gen on resume
        warnings.append(
            f"frame {order}: image generation failed "
            f"({failures.get(order, 'unknown')}); {how} to avoid a gap"
        )
        frames.append(FrameMeta(order=order, path=path, caption=shot.narration, status="degraded"))

    # Mark the STAGE degraded (not the job) so the front end shows it amber and the
    # wrapper's stage_finish no-ops. The runner still derives the job's
    # completed_with_warnings status from the degraded frames in _persist.
    if warnings and reporter is not None:
        await reporter.mark_degraded(
            f"{len(warnings)} of {total} frame(s) degraded (neighbour-filled / placeholder)"
        )

    return {"frames": frames, "warnings": warnings}
