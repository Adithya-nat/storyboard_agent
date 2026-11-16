# Creator Agent

Open-source AI film studio for creators. Write one line and get a post-ready vertical short film
(30 or 60s): an original cast, AI-filmed scenes (Gemini Omni) where the characters speak their
lines on camera, a narrator and/or music score, and per-platform post copy. Each film is entirely
English or entirely Tamil.

- **Homepage:** <https://adithya-nat.github.io/creator-agent/>
- **Web app:** [creator-agent-web](https://github.com/Adithya-nat/creator-agent-web)
- **Bring your own key:** every film is made with the user's own Gemini API key.
- **24-hour retention:** everything a film makes is deleted 24 hours after it starts.

**Run it yourself:** start with [SETUP.md](SETUP.md) (keys, spend limits, a setup checker).
**Host it for others:** follow [DEPLOY.md](DEPLOY.md) (Neon, sign-in, a free server, the homepage).

## Run it yourself

```bash
git clone https://github.com/Adithya-nat/creator-agent
git clone https://github.com/Adithya-nat/creator-agent-web
cd creator-agent
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

## Pause, resume and breakpoints

Pausing never wastes a paid call. Once you pause, no new frame, scene clip, reference
sheet or narration line starts. Calls already submitted finish and are kept, since
they're billed either way. Then the job parks as `paused`. Every finished unit is on
disk, and every stage skips what's on disk, so resuming continues from the first
unfinished unit: if scene 3 of 6 was still waiting to start, the resume starts at
scene 3. Resuming while a pause is still draining cancels the pause. Breakpoints pause
right before a stage, for example to read the screenplay before paying for AI video.
Paused films follow the 24-hour retention rule like any film; a self-hosted install can
set `PAUSED_TTL_SECONDS` to keep a paused film longer, counted from when it paused.

Tamil films stay Tamil on screen. The app UI stays English and shows an English
rendering under each Tamil title and line: one cheap, cached text call (`gloss.json`).

`make diagram` prints the graph as Mermaid straight from `app/graph/build.py`.

## Story styles and languages

- **Dialogue** (default): the cast acts and speaks every scene on camera.
- **Mixed**: the narrator sets up and closes the story; the inciting beat and climax cut to dialogue.
- **Narrated**: the channel narrator tells the story over AI-filmed scenes.
- **Languages**: `en` or `ta`. A Tamil film is Tamil in speech, captions and post copy.

## API

```
POST /storyboard                      {"prompt", "duration": 30|60, "style", "language"} → job_id
GET  /storyboard                      your films from the last 24 hours
GET  /storyboard/{id}                 status, current stage, per-stage progress, cost
GET  /storyboard/{id}/events          SSE: live status, stage and log stream
GET  /storyboard/{id}/logs?stage=     per-stage log lines (paginated)
GET  /storyboard/{id}/storyboard      script, cast, scenes, cover, cost
GET  /storyboard/{id}/publish         post copy per platform
GET  /storyboard/{id}/video           the mp4
POST /storyboard/{id}/resume          continue a stopped/failed/paused job (finished work is reused)
POST /storyboard/{id}/pause           stop starting paid work; in-flight units finish, then it parks
PUT  /storyboard/{id}/breakpoints     {"stages": [...]}: pause right BEFORE these stages
GET  /storyboard/{id}/checkpoint      per-unit state (done / in flight / pending) + cost to finish
GET  /budget  ·  /estimate  ·  /usage/summary  ·  /usage/runs  ·  /health
GET  /me                              who is signed in, beta access, media-link token
POST /keys/check                      which models a Gemini key can use (free)
```

In hosted mode (`AUTH_MODE=jwt`) every route needs `Authorization: Bearer <token>` from the web
app's sign-in. Video, images and the live stream also accept a signed `?t=` link, because
`<video>` and `EventSource` can't send headers. Create and resume take the user's key in
`X-Gemini-Key` (`KEY_MODE=byok`).

```bash
curl -s -XPOST localhost:8000/storyboard -H 'content-type: application/json' \
  -d '{"prompt":"a grandmother teaches her grandson to fly a kite","duration":30,"style":"dialogue","language":"en"}'
```

Job statuses: `queued → running → completed | completed_with_warnings | failed | interrupted`.
If the API restarts mid-job, the job is marked `interrupted`; resume continues it.

## Hosted mode: sign-in, your own key, 24 hours

| Setting | Local (default) | Hosted beta |
| --- | --- | --- |
| `AUTH_MODE` | `none`: one local user | `jwt`: Better Auth tokens from the web app, checked against its JWKS |
| `KEY_MODE` | `server`: `GEMINI_API_KEY` in `.env` | `byok`: each request carries the user's key |
| `ALLOWED_USERS` | unused | emails or `@domain.com` allowed to make films (beta allowlist) |
| Budget | `MONTHLY_BUDGET_USD` for everything | `USER_MONTHLY_BUDGET_USD` per user |

- **The key is never stored.** It lives in the user's browser, rides on the create/resume request,
  is held in memory for that film only, and is redacted from logs and error messages.
- **Rate limits follow the key.** Each key gets its own limiter and spend governor, matching
  Google's per-project quotas.
- **Per-user data.** Jobs, usage and budget are scoped to the signed-in user; someone else's
  job is a 404.
- **24-hour retention.** The sweeper wakes when the next job expires and deletes its rows, files,
  graph checkpoints and the prompt in the usage ledger. Only cost totals remain.

## Spend and safety rails

- **Budget guard**: new films are refused (HTTP 402) when month-to-date spend plus the estimate
  would pass the cap (`MONTHLY_BUDGET_USD`, default $300; per user when hosted).
- **Spend-rate governor**: paces video and image calls under Google's rolling 10-minute spend limit.
- **No double billing**: each paid Omni submission is journalled; a crash or timeout never resubmits it.
- **Usage ledger**: every call's cost is recorded in `spend_events` and `run_ledger`. They outlive
  the 24-hour sweep (without the prompt), so the Usage page keeps cost history.
- **Mock mode never spends**: with `USE_MOCK_PROVIDERS=true` no request reaches Google, even if a
  key is set.

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
site/            the homepage (GitHub Pages, deployed by .github/workflows/pages.yml)
deploy/          Dockerfiles, compose file and Caddy config for a self-hosted server
evals/           golden-prompt eval harness (make evals / make evals-mock)
tests/           offline tests (mock providers); `-m live` for real-API smoke tests
```

Configuration lives in `.env`; see [SETUP.md](SETUP.md) for the settings that matter.
