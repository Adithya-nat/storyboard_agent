<!-- The blocked scene to rewrite. Vars: code, message, attempt_note, mode, language, scene, lines, characters, shots_rule. -->
Blocked with code "{{ code }}": {{ message }}
{{ attempt_note }}
Scene mode: {{ mode }} (language: {{ language }})
Characters in frame: {{ characters }}

Original scene description:
{{ scene }}

Original dialogue lines (speaker: text (delivery)):
{{ lines }}

{{ shots_rule }}

Return JSON with: scene (the rewritten scene description), image_prompt, lines (same speakers in order; empty when there are none), shots, change.
