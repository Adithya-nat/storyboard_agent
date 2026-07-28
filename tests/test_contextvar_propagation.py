"""The load-bearing correctness test for log attribution.

image_gen runs per-frame work via asyncio.gather, and each frame's image call via
asyncio.to_thread. For the stdlib bridge to tag the adapter's re-roll logs with
the image_gen stage, the (job_id, stage) contextvar bound at the node level MUST
be visible inside BOTH a gather child AND a thread spawned from it. Python copies
the context in both cases — this test pins that guarantee so a future refactor
can't silently break stage attribution.
"""

from __future__ import annotations

import asyncio

from app.observability.context import bind_stage, current, reset_stage


async def test_contextvar_visible_across_gather_and_to_thread():
    seen: dict[str, tuple] = {}

    def _in_thread():
        ctx = current()
        return None if ctx is None else (ctx.job_id, ctx.stage)

    async def worker(key: str):
        c = current()  # visible in the gather child coroutine
        t = await asyncio.to_thread(_in_thread)  # ...and in the thread it spawns
        seen[key] = (None if c is None else (c.job_id, c.stage), t)

    token = bind_stage("job-x", "image_gen")
    try:
        await asyncio.gather(worker("a"), worker("b"))
    finally:
        reset_stage(token)

    for key in ("a", "b"):
        coro_ctx, thread_ctx = seen[key]
        assert coro_ctx == ("job-x", "image_gen")
        assert thread_ctx == ("job-x", "image_gen")


async def test_reset_clears_binding():
    token = bind_stage("j", "designer")
    ctx = current()
    assert ctx is not None and (ctx.job_id, ctx.stage) == ("j", "designer")
    reset_stage(token)
    assert current() is None
