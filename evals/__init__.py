"""Manual golden-prompt eval harness (NOT run in CI).

Mocked unit tests prove the pipeline *wires up*; they can't measure whether the
script reads well, the style holds, or captions are the right length. This harness
runs a handful of golden prompts against the REAL models and dumps every artifact
for human review, with cheap automatic heuristics (schema, frame count, caption
length) to flag obvious regressions.
"""
