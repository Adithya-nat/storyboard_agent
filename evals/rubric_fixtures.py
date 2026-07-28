"""Known-good / known-bad scripts to validate the narrative judge's rubric.

The judge is only useful if its score CORRELATES with the real complaint — a coherent
story should score well above an incoherent pile of pretty shots. These fixtures let you
prove that with a real key:

    python -m evals.rubric_fixtures      # needs GEMINI_API_KEY

In MOCK mode the judge returns a fixed score (no signal), so the comparison is only
meaningful against the real model. The unit tests assert the fixtures are well-formed;
this script asserts good > bad when run for real.
"""

from __future__ import annotations

from app.agents.judge import NarrativeJudge
from app.core.config import get_settings
from app.schemas import Scene, Script

# A real story: setup -> rising -> climax -> resolution, one through-line, one world.
GOOD_SCRIPT = Script(
    title="The Lighthouse and the Petrel",
    logline="A lonely keeper nurses a storm-blown petrel back to flight over one winter.",
    style="muted cinematic, cold blues warming to gold",
    arc=(
        "Beginning: an isolated lighthouse keeper finds a hurt storm petrel after a gale. "
        "Middle: through the winter he shelters and feeds it, and the routine thaws his "
        "loneliness. End: at the first spring light he frees it, and it circles back once "
        "before flying on — he is alone again, but no longer lonely."
    ),
    scenes=[
        Scene(order=1, beat_role="setup",
              description="A solitary lighthouse on a black winter cliff, one lit window.",
              narration="At the edge of the world, one keeper, one light."),
        Scene(order=2, beat_role="inciting",
              description="The keeper kneels over a storm-battered petrel on wet rocks.",
              narration="The gale leaves him a small, broken visitor."),
        Scene(order=3, beat_role="rising",
              description="Inside, the keeper feeds the petrel by lamplight, snow at the glass.",
              narration="All winter, he keeps it warm, and it keeps him company."),
        Scene(order=4, beat_role="climax",
              description="Spring dawn; the keeper lifts the healed petrel to an open sky.",
              narration="When the light returns, he opens his hands."),
        Scene(order=5, beat_role="resolution",
              description="The petrel circles the lighthouse once, then flies to sea; he watches, calm.",
              narration="It circles once, in thanks, and is gone — and he is at peace."),
    ],
)

# A non-story: disconnected pretty shots, no causality, no arc, unresolved.
BAD_SCRIPT = Script(
    title="Vibes",
    logline="Cool images.",
    style="neon, epic, cinematic",
    arc="",
    scenes=[
        Scene(order=1, beat_role="",
              description="A neon city at night, glowing orbs everywhere.",
              narration="In a world of light."),
        Scene(order=2, beat_role="",
              description="A different desert with a glowing tree, no characters.",
              narration="Beauty everywhere."),
        Scene(order=3, beat_role="",
              description="An unrelated astronaut underwater for no reason.",
              narration="So epic."),
        Scene(order=4, beat_role="",
              description="A brand-new dragon appears, then nothing resolves.",
              narration="Wow."),
        Scene(order=5, beat_role="",
              description="Random close-up of an eye, unrelated to anything prior.",
              narration="The end?"),
    ],
)


def main() -> None:
    judge = NarrativeJudge(settings=get_settings())
    good = judge.score(GOOD_SCRIPT).overall
    bad = judge.score(BAD_SCRIPT).overall
    print(f"GOOD overall = {good}\nBAD  overall = {bad}")
    assert good > bad, f"rubric broken: good({good}) !> bad({bad})"
    print("OK: judge ranks the coherent story above the incoherent one.")


if __name__ == "__main__":
    main()
