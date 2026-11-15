<!-- Attribute a dialogue clip's speech to the cast. Vars: speakers, script. -->
This audio is one scene of a short film. The characters are: {{ speakers }}.
The script for this scene was:
{{ script }}

Listen and return what was ACTUALLY said, in order, as JSON lines of {speaker, text}. Use exactly the character names above. Keep the original language and script. If a line was said differently from the script, return what was said; if a line was not said, omit it.
