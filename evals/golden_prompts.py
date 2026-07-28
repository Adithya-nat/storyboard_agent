"""Golden prompts for the manual eval harness.

A small, deliberately varied set: different genres, subjects, and both durations,
so style consistency and frame scaling are exercised across cases.
"""

from __future__ import annotations

# (prompt, duration_seconds)
GOLDEN_PROMPTS: list[tuple[str, int]] = [
    ("A lonely lighthouse keeper befriends a storm petrel over one winter", 30),
    ("The last bookstore on Mars, told through its closing night", 60),
    ("A street cat's secret midnight tour of a sleeping Tokyo", 30),
    ("Two rival tea masters settle their feud in a single ceremony", 60),
    ("A child's crayon drawing comes alive and redecorates the house", 30),
]
