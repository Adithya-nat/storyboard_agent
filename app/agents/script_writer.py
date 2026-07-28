"""script_writer agent: user prompt -> structured Script (scenes + narration).

This is the first creative stage. It turns a one-line idea into an ordered set
of visual beats, each with a "what we see" description (drives the image model)
and narration text (caption now, TTS-ready later). Output is schema-validated
and saved as an artifact; downstream nodes load it by reference.
"""

from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState
from app.schemas import Scene, Script

log = get_logger(__name__)

# Structural roles a beat can play in the arc. Order matters: it's the narrative
# spine the prompt + judge + reflection all reference.
BEAT_ROLES = ("setup", "inciting", "rising", "climax", "resolution")


def _system_prompt(n: int) -> str:
    """Build the script_writer system prompt for an ``n``-beat film.

    Built as a function (not a module ``str.format`` constant) ON PURPOSE: the prompt
    embeds nothing brace-shaped, but interpolating ``n`` via an f-string sidesteps the
    ``str.format`` brace trap entirely if examples are ever added later.

    Technique stack: role priming, an explicit 3-act scaffold mapped onto the N beats,
    plan-then-write ordering (write ``arc`` first, then scenes that deliver it), and a
    DO/DON'T block of negative constraints — the levers that turn "N pretty shots" into
    one coherent micro-story with a real start, middle, and end.
    """
    return (
        "<role>\n"
        "You are an award-winning short-form film director and screenwriter. Your job "
        "is NOT pretty pictures — it is ONE coherent micro-story a viewer instantly "
        "understands and FEELS, with a clear beginning, middle, and end.\n"
        "</role>\n\n"
        "<task>\n"
        f"Turn the user's idea into a storyboard of EXACTLY {n} scenes. Think of a "
        f"three-act story compressed into {n} beats:\n"
        "  - BEGINNING (~first fifth): establish the main character, the place, and the "
        "hook — what is normal and what desire or question pulls us in.\n"
        "  - MIDDLE (~middle three-fifths): develop it. Each beat is a CONSEQUENCE of "
        "the one before — rising tension, a complication, a turn. Never a loose list of "
        "moments.\n"
        "  - END (~last fifth): the climax and a clear resolution that answers the hook. "
        "The final beat must FEEL like an ending, not a random shot.\n"
        "</task>\n\n"
        "<process>\n"
        "Work in THIS order:\n"
        "  1. Write 'arc': one short paragraph stating the beginning -> middle -> end in "
        "plain language (who, what they want, what changes, how it resolves).\n"
        "  2. Write 'scenes' that DELIVER that arc. For each scene set 'beat_role' to one "
        f"of: {', '.join(BEAT_ROLES)}. 'description' = what the CAMERA SEES (concrete, "
        "filmable, the SAME character(s) and setting kept identical across scenes). "
        "'narration' = one short caption sentence (aim for under ~15 words) that MOVES "
        "THE STORY FORWARD.\n"
        "</process>\n\n"
        "<rules>\n"
        "DO:\n"
        "  - Keep one continuous through-line and one consistent visual world.\n"
        "  - Make every beat follow causally from the previous one.\n"
        "  - Give the story a real turn and a satisfying payoff.\n"
        "  - Keep characters, names, and setting identical across every scene.\n"
        "DON'T:\n"
        "  - Don't emit disconnected 'pretty' shots or a vibe montage with no cause/effect.\n"
        f"  - Don't just restate or re-describe the prompt {n} times.\n"
        "  - Don't leave the ending unresolved or introduce a brand-new subject in the "
        "last beat.\n"
        "  - Don't use AI cliches (neon everything, 'in a world...', glowing orbs for no "
        "reason).\n"
        "</rules>\n\n"
        "<output>\n"
        f"Return STRICT JSON matching the Script schema. EXACTLY {n} scenes, 'order' "
        f"1..{n} in story order. No prose outside the JSON.\n"
        "</output>"
    )


def _beat_roles(n: int) -> list[str]:
    """Assign a structural role to each of ``n`` beats (a sane default + mock spine).

    setup -> inciting -> rising... -> climax -> resolution, scaled to the beat count.
    """
    if n <= 1:
        return ["setup"]
    roles: list[str] = []
    for i in range(n):
        frac = i / (n - 1)
        if i == 0:
            roles.append("setup")
        elif i == 1 and n >= 5:
            roles.append("inciting")
        elif frac >= 0.88:
            roles.append("resolution")
        elif frac >= 0.72:
            roles.append("climax")
        else:
            roles.append("rising")
    return roles


def _mock_script(prompt: str, n: int) -> Script:
    """Deterministic offline script so the pipeline runs without an API key."""
    roles = _beat_roles(n)
    return Script(
        title=f"Mock: {prompt[:40]}",
        logline=prompt,
        style="clean cinematic, consistent palette (mock)",
        arc=(
            f"Beginning: we meet {prompt[:48]} and what they want. Middle: it develops "
            "and complicates beat by beat. End: it resolves with a clear payoff."
        ),
        scenes=[
            Scene(
                order=i,
                beat_role=roles[i - 1],
                description=f"Scene {i}: {prompt} — beat {i} of {n}.",
                narration=f"Beat {i}: {prompt[:60]}",
            )
            for i in range(1, n + 1)
        ],
    )


async def script_writer(state: StoryboardState, deps: Deps) -> StoryboardState:
    # Idempotency (resume): reuse an existing script verbatim. The LLM is
    # non-deterministic, so re-running it would produce a DIFFERENT script that
    # no longer matches an already-built shotlist/frames. Load-or-produce keeps
    # resume correct (and skips a wasted LLM call).
    if deps.store.exists("script.json"):
        log.info("reusing the existing script (resume)")
        return {"script_path": deps.store.abspath("script.json")}

    n = deps.frame_count(state["duration"])  # 1 frame / seconds_per_frame
    log.info("writing a %d-beat script from the prompt: %r", n, state["prompt"][:80])
    script: Script = await asyncio.to_thread(
        deps.llm.complete_json,
        system=_system_prompt(n),
        user=(
            f"<idea>{state['prompt']}</idea>\n"
            f"Target length: {state['duration']}s, {n} scenes. The tagged idea is the "
            "creative brief to dramatize."
        ),
        response_model=Script,
        mock_factory=lambda _u: _mock_script(state["prompt"], n),
    )
    path = deps.store.save_json("script.json", script)
    log.info("script ready: %r — %d scenes", script.title, len(script.scenes))
    return {"script_path": path}
