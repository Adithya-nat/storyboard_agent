"""Print the LangGraph pipeline as Mermaid, straight from the real graph.

Regenerate the diagram in README.md after changing the node order in
app/graph/build.py:

    python -m scripts.render_pipeline_diagram
"""

from __future__ import annotations

from types import SimpleNamespace

from app.core.config import Settings
from app.graph.build import build_graph
from app.graph.state import Deps


def main() -> None:
    deps = Deps(
        settings=Settings(motion_engine_enabled=True, use_mock_providers=True),
        store=SimpleNamespace(),
        llm=SimpleNamespace(),
        image=SimpleNamespace(),
        tts=None,
        reporter=None,
        motion=None,
    )
    graph = build_graph(deps)
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
