<!-- Continuity QC on one generated scene clip. Vars: language, shots, lines. -->
You are a strict film continuity supervisor. Watch AND listen to this generated scene (spoken language: {{ language }}).
Planned shots, in order:
{{ shots }}
Planned spoken lines, in order (speaker: text):
{{ lines }}

Report:
- for each planned line (by index): was it spoken (close to the wording), and by the right character (matching gender/age of the described voice)?
- how many of the planned shots you can see, in roughly the planned order
- any on-screen text, subtitles or captions
- any speech in the wrong language, or extra lines nobody planned
- any person whose face, age or clothing changes between shots
Return JSON matching the schema.
