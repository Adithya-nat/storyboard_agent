<!-- The film and the creator's request. Vars: note, picked, title, logline, language, characters, scenes, history. -->
<creator_note>
{{ note }}
</creator_note>

<picked>
{{ picked }}
</picked>

<film title="{{ title }}" language="{{ language }}">
Logline: {{ logline }}
Characters: {{ characters }}
</film>

<scenes>
{{ scenes }}
</scenes>

<history>
{{ history }}
</history>

Return JSON: summary, scenes (one entry per picked scene you change: order, description, image_prompt, narration for narrated scenes, shots with index/action/size/camera/lines), story_note, look_note, score_note, release_note, extra.
