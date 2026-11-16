"""Exit 1 if any film is queued, running or pausing (used by `make host-up` before a restart)."""

from __future__ import annotations

import asyncio
import sys

from app.db import repository as repo
from app.db.session import get_sessionmaker


async def _count() -> int:
    async with get_sessionmaker()() as session:
        return await repo.count_active_jobs(session)


if __name__ == "__main__":
    n = asyncio.run(_count())
    if n:
        print(f"{n} film(s) are rendering; restarting now would interrupt them. Wait, or run with FORCE=1.")
        sys.exit(1)
    print("no films rendering")
