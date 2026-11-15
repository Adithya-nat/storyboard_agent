# Storyboard Agent

Turns one line of text into a post-ready vertical short film (30 or 60s): an original cast,
AI-filmed scenes (Gemini Omni) where the characters speak their lines on camera, a narrator
and/or music score, and per-platform post copy. Each film is entirely English or entirely Tamil.

**New here? Start with [SETUP.md](SETUP.md)**: keys, spend limits and a setup checker.

## Quick start

```bash
make setup     # install everything, create .env files, start Postgres, migrate
# add GEMINI_API_KEY to .env
make doctor    # checks keys, models, ffmpeg, Tamil fonts, database
make dev       # API on :8000 + web app on :3000 (Ctrl+C stops both)
```

Then open <http://localhost:3000>. Run `make` to see every command.

| Command | What it does |
| --- | --- |
| `make dev` / `make stop` / `make status` | Start, stop or check the API and web app |
| `make api` / `make web` | Start just one of them |
| `make health` / `make budget` | Model check and this month's spend |
| `make voice DESCRIBE="..." GENDER=male VOICE_LANG=en-IN` | Design the channel narrator voice |
| `make test` / `make check` | API tests / everything that must pass before a commit |
| `make doctor-live` / `make test-live` | Real-API checks (~$0.20 / ~$0.50) |
| `make db-shell` / `make migrate` | Postgres shell / apply migrations |
| `make clean` | Remove caches and the web build |

## How a film is made

One LangGraph `StateGraph`, the same for every story style. Stages that don't apply to a film's
style report **skipped**.

```
brief → script_writer → dialogue_pass → reflection → designer → image_gen → narration → motion → captions → music → assembler → film_review → publish_kit
```

| Stage | Model | Output |
| --- | --- | --- |
| `brief` | Gemini 4 Carbon | Creative brief: format, purpose, 2-second hook, setting, era and speech register, scene plan (keeps any structure the idea spells out) |
| `script_writer` | Gemini 4 Carbon | Screenplay: 2-12 scenes of 5-15s, each covered by 2-5 shots (silent shots allowed) with the lines spoken in them |
| `dialogue_pass` | Gemini 4 Carbon | Rewrites every line of the film in one pass so it flows as conversation, in today's slang or period speech (`app/prompts/register/`) |
| `reflection` | Gemini 4 Argon judge + Carbon reviser | Judges hook, brief, structure, continuity, dialogue thread and register; revises until ≥ 90 (max 2 rounds) |
| `designer` | Gemini 4 Argon + Nano Banana Pro | Style bible, style reference, 3-view sheet per character, one reference plate per location |
| `image_gen` | Nano Banana Pro | One opening frame per scene from labelled cast + location references, with an identity check |
| `narration` | Gemini 3.8 Flash TTS | Narrator lines (narrated scenes only) |
| `motion` | Gemini Omni 1.2 Flash | One 1080×1920 multi-shot clip per scene (timecoded cuts, same set and voices), dialogue spoken on camera; a QC watch re-renders a failed scene once |
| `captions` | Gemini 3.5 Transcribe | Speaker-labelled subtitles (only when `BURN_SUBTITLES=true`) |
| `music` | Lyria 3.5 | One instrumental score for the whole film |
| `assembler` | ffmpeg | Hard cuts between scenes (dissolve only for time jumps), speech + ambience + score sidechain-ducked under speech |
| `film_review` | Gemini 4 Argon | Watches the finished film as a first-time viewer; coherence score on the result page and in the usage ledger |
| `publish_kit` | Gemini 4 Argon | Title, hook, captions and hashtags for TikTok, Reels and Shorts |

Every scene must be AI video. If Omni can't produce a clip (for example, the daily quota is
used up), the job stops instead of shipping stills; `POST /resume` finishes it later and reuses
every clip already paid for.

`make diagram` prints the graph as Mermaid straight from `app/graph/build.py`.

## Story styles and languages

- **Dialogue** (default): the cast acts and speaks every scene on camera.
- **Mixed**: the narrator sets up and closes the story; the inciting beat and climax cut to dialogue.
- **Narrated**: the channel narrator tells the story over AI-filmed scenes.
- **Languages**: `en` or `ta`. A Tamil film is Tamil in speech, captions and post copy.

## API

```
POST /storyboard                      {"prompt", "duration": 30|60, "style", "language"} → job_id
GET  /storyboard/{id}                 status, current stage, per-stage progress, cost
GET  /storyboard/{id}/events          SSE: live status, stage and log stream
GET  /storyboard/{id}/logs?stage=     per-stage log lines (paginated)
GET  /storyboard/{id}/storyboard      script, cast, scenes, cover, cost
GET  /storyboard/{id}/publish         post copy per platform
GET  /storyboard/{id}/video           the mp4
POST /storyboard/{id}/resume          continue a stopped/failed job (finished work is reused)
GET  /budget  ·  /estimate  ·  /usage/summary  ·  /usage/runs  ·  /health
```

```bash
curl -s -XPOST localhost:8000/storyboard -H 'content-type: application/json' \
  -d '{"prompt":"a grandmother teaches her grandson to fly a kite","duration":30,"style":"dialogue","language":"en"}'
```

Job statuses: `queued → running → completed | completed_with_warnings | failed | interrupted`.
If the API restarts mid-job, the job is marked `interrupted`; resume continues it.

## Spend and safety rails

- **Budget guard**: new films are refused (HTTP 402) when month-to-date spend plus the estimate
  would pass `MONTHLY_BUDGET_USD` (default $300).
- **Spend-rate governor**: paces video and image calls under Google's rolling 10-minute spend limit.
- **No double billing**: each paid Omni submission is journalled; a crash or timeout never resubmits it.
- **Usage ledger**: every call's cost is recorded in `spend_events` and `run_ledger`, which are never
  swept, so the web app's Usage page keeps history after films expire.

## Project layout

```
app/
  agents/        one file per pipeline stage
  prompts/       every agent prompt, one file each ({{ var }} templates)
  adapters/      Gemini text, images, Omni video, TTS, transcribe, Lyria, ffmpeg, rate limits
  graph/         LangGraph state, wiring, checkpointer
  observability/ cost table, spend events, budget, spend-rate governor, stage reporter
  api/ db/ workers/ core/
scripts/         doctor, design_voice, probes and bake-offs
evals/           golden-prompt eval harness (make evals / make evals-mock)
tests/           offline tests (mock providers); `-m live` for real-API smoke tests
```

Configuration lives in `.env`; see [SETUP.md](SETUP.md) for the settings that matter.
