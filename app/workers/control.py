"""Run controls: instant pause, breakpoints and the in-flight unit registry.

A pause takes effect at once; the user never waits for the current step:

* The runner cancels the graph as soon as a pause is requested. Text work in progress
  (an LLM call) is dropped and redone on resume; it costs a fraction of a cent.
* A paid media call that was already submitted (a frame, a scene clip, a reference sheet,
  a narration line) is billed whether we wait or not, since it runs in a worker thread
  that can't be stopped. Each one runs under ``keep()``, so cancelling the stage doesn't
  lose it: it finishes in the background and saves its own output. A resume waits for
  those calls (``settle()``) before it starts, so nothing is paid for twice.
* No NEW paid unit starts once a pause is requested: every unit passes ``gate()`` right
  before it spends.
* Every finished unit is on disk, and every stage skips units that are on disk. So a
  resume continues from the first unit that isn't done: scene 3 of 6 resumes at scene 3.

The pause flag lives on the job row (``pause_requested_at``), so it is visible across
processes, and is mirrored in an in-process event so the runner reacts instantly.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Awaitable, Callable, Iterator, TypeVar

from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = get_logger(__name__)
T = TypeVar("T")


class PauseRequested(Exception):
    """Raised by a unit gate: a pause was requested, so this paid unit must not start."""


class JobPaused(Exception):
    """Raised out of a stage once its in-flight units have drained; the runner parks the job."""

    def __init__(self, stage: str | None, message: str, *, breakpoint: bool = False) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.breakpoint = breakpoint


# --------------------------------------------------------------------------- #
# In-process registry (one API process runs the jobs in BackgroundTasks).
# --------------------------------------------------------------------------- #
_SIGNALS: dict[str, asyncio.Event] = {}
_INFLIGHT: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
_KEPT: dict[str, set[asyncio.Task]] = defaultdict(set)  # paid calls that outlive a pause


def _event(job_id: str) -> asyncio.Event:
    ev = _SIGNALS.get(job_id)
    if ev is None:
        ev = _SIGNALS[job_id] = asyncio.Event()
    return ev


def signal_pause(job_id: str) -> None:
    _event(job_id).set()


def clear_pause(job_id: str) -> None:
    _event(job_id).clear()


def in_flight(job_id: str) -> dict[str, list[int]]:
    """Units currently being generated (and paid for), per stage."""
    stages = _INFLIGHT.get(job_id) or {}
    return {name: sorted(units) for name, units in stages.items() if units}


def forget(job_id: str) -> None:
    _SIGNALS.pop(job_id, None)
    if not _KEPT.get(job_id):  # paid calls still landing keep showing as in flight
        _INFLIGHT.pop(job_id, None)
        _KEPT.pop(job_id, None)


async def settle(job_id: str) -> None:
    """Wait for paid calls a pause left running, so their output is on disk."""
    while tasks := list(_KEPT.get(job_id) or ()):
        await asyncio.gather(*tasks, return_exceptions=True)


def _kept_done(job_id: str, task: asyncio.Task) -> None:
    _KEPT[job_id].discard(task)
    if not task.cancelled() and (exc := task.exception()) is not None and not isinstance(exc, PauseRequested):
        log.info("job %s: a paid call finished after the pause with %s", job_id, type(exc).__name__)


class RunControl:
    """The runner's handle on pause state for one job. Nodes reach it via ``deps.control``."""

    def __init__(
        self,
        job_id: str,
        sessionmaker: "async_sessionmaker[AsyncSession] | None" = None,
        *,
        poll_seconds: float = 2.0,
    ) -> None:
        self.job_id = job_id
        self._sm = sessionmaker
        self._poll = poll_seconds
        self._checked = 0.0
        self._db_flag = False
        self.stage: str | None = None  # the stage running now (set by the graph wrapper)
        self.held: tuple[str, int, str] | None = None  # (stage, seq, message) where a pause stopped it

    async def requested(self) -> bool:
        """True once a pause was requested. The in-process event answers immediately;
        the DB is consulted at most every ``poll_seconds`` (another process may own it)."""
        if _event(self.job_id).is_set():
            return True
        if self._sm is None:
            return False
        now = time.monotonic()
        if now - self._checked >= self._poll:
            self._checked = now
            from app.db import repository as repo

            try:
                async with self._sm() as s:
                    self._db_flag = await repo.pause_requested(s, self.job_id)
            except Exception:  # noqa: BLE001 - a flaky read never stops a film
                log.warning("pause flag read failed for job %s", self.job_id, exc_info=True)
            if self._db_flag:
                signal_pause(self.job_id)
        # A pause signalled while the DB read was in flight still counts.
        return self._db_flag or _event(self.job_id).is_set()

    def reset(self) -> None:
        """Forget the cached DB answer (a withdrawn pause): the next check reads it afresh."""
        self._db_flag = False
        self._checked = 0.0
        self.held = None

    async def gate(self) -> None:
        if await self.requested():
            raise PauseRequested()

    async def until_requested(self) -> None:
        """Return once a pause is requested (the runner's watcher)."""
        while not await self.wait(3600):
            pass

    async def keep(self, work: Callable[[], Awaitable[T]], stage: str | None = None, unit: int | None = None) -> T:
        """Run one paid call so cancelling the stage can't throw it away (see module doc)."""

        async def run() -> T:
            with self.unit(stage, unit) if stage is not None and unit is not None else nullcontext():
                return await work()

        task = asyncio.ensure_future(run())
        _KEPT[self.job_id].add(task)
        task.add_done_callback(lambda t: _kept_done(self.job_id, t))
        return await asyncio.shield(task)

    async def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, waking early on a pause request. True if paused.

        Waits in ``poll_seconds`` slices so a pause recorded by another process (DB only,
        no local event) still cuts the wait short."""
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return await self.requested()
            try:
                await asyncio.wait_for(_event(self.job_id).wait(), timeout=min(left, self._poll))
                return True
            except asyncio.TimeoutError:
                if await self.requested():
                    return True

    async def take_breakpoint(self, stage: str) -> bool:
        """If ``stage`` is a breakpoint, consume it and raise the pause flag. A consumed
        breakpoint doesn't fire again, so resuming carries on into the stage."""
        if self._sm is None:
            return False
        from app.db import repository as repo

        async with self._sm() as s:
            hit = await repo.consume_breakpoint(s, self.job_id, stage)
        if hit:
            signal_pause(self.job_id)
        return hit

    @contextmanager
    def unit(self, stage: str, unit: int) -> Iterator[None]:
        """Mark one unit as in flight for the checkpoint view."""
        _INFLIGHT[self.job_id][stage].add(unit)
        try:
            yield
        finally:
            _INFLIGHT[self.job_id][stage].discard(unit)


# --------------------------------------------------------------------------- #
# Node helpers: no-ops when the node runs without a control (unit tests, evals).
# --------------------------------------------------------------------------- #
async def gate(deps) -> None:
    control = getattr(deps, "control", None)
    if control is not None:
        await control.gate()


async def keep(deps, work: Callable[[], Awaitable[T]], stage: str | None = None, unit: int | None = None) -> T:
    """Run a paid call (provider call + saving its output) so a pause can't lose it."""
    control = getattr(deps, "control", None)
    if control is None:
        return await work()
    return await control.keep(work, stage, unit)


def track(deps, stage: str, unit: int):
    control = getattr(deps, "control", None)
    return control.unit(stage, unit) if control is not None else nullcontext()


async def interruptible_sleep(deps, seconds: float) -> bool:
    """``asyncio.sleep`` that a pause cuts short. True if a pause interrupted it."""
    control = getattr(deps, "control", None)
    if control is None:
        await asyncio.sleep(seconds)
        return False
    return await control.wait(seconds)


def paused(stage: str, done: int, total: int, noun: str) -> JobPaused:
    left = max(0, total - done)
    return JobPaused(stage, f"Paused after {done} of {total} {noun}; {left} left")
