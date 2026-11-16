# Setup

Everything you need to go from a fresh machine to posting AI short films. Covers all three repos.
Every step has a `make` shortcut (run `make` in `creator-agent` to list them all).
This guide runs Creator Agent on your own machine. To host it for other people, see [DEPLOY.md](DEPLOY.md).

| Repo | What it is | Port |
| --- | --- | --- |
| `creator-agent` (this repo) | API + the AI pipeline (script, cast, frames, AI video, voices, music, edit) | 8000 |
| `creator-agent-web` | The web app | 3000 |
| `storyboard_engine` | Optional legacy local-motion service. Not needed: AI video runs inside the API. | 8090 |

## 1. Install the tools

```bash
brew install python@3.12 node ffmpeg libraqm   # libraqm = correct Tamil text in captions
brew install --cask docker                     # Postgres runs in Docker
```

`ffmpeg` must include `zoompan`, `xfade` and `sidechaincompress` (Homebrew's build does).

## 2. Get your API keys

**Gemini API key (required).** Every model (script, images, AI video, voices, music) runs on it.

1. Open [AI Studio → API keys](https://aistudio.google.com/apikey) and create a key in a project with billing turned on.
2. Billing matters: free-tier keys can't use Gemini Omni (AI video) or Nano Banana Pro.

**Anthropic API key (optional).** Only used if you switch the script writer to Claude (see `SCRIPT_MODEL` below). Create one in the [Anthropic Console](https://console.anthropic.com/settings/keys).

**Cloudflare Turnstile keys (optional).** Only if you put the app on the public internet, to stop strangers spending your credits.

## 3. Understand the spend limits before your first video

- The app stops itself at **`MONTHLY_BUDGET_USD`** (default `300`). The Usage page shows spend per run and how many videos you have left.
- Google also enforces a **billing-account cap per tier**. On Tier 1 it's **$250/month**: when you reach it, Google pauses *every* project on that billing account until the 1st. You move to Tier 2 ($2,000 cap) automatically after $100 total spend and 3 days. Until then, keep `MONTHLY_BUDGET_USD` at `240` or lower.
- As a hard backstop, set a project spend cap in AI Studio (Billing → Spend cap) to the same value.
- Google also limits **how fast** you spend: **$10 per rolling 10 minutes on Tier 1** ($200 from Tier 2). The app paces AI-video and image calls to stay under `SPEND_RATE_LIMIT_USD_PER_10MIN` (default `9`), so a 60s film simply takes a few minutes longer on Tier 1. Raise it to `190` once you reach Tier 2.
- To top up, add credits in AI Studio (Billing). Prepaid accounts fail with HTTP 402 at $0, and the app shows that error on the create page.

Typical cost (measured on real films, Nov 2026): **~$6–8 for a 30s film, ~$13–15 for 60s**. About two thirds is AI video (~$0.10 per second at 1080p, plus a voice re-render now and then); most of the rest is images (character sheets and one 2K frame per scene).

## 4. Configure the API

```bash
cd creator-agent
make setup    # .venv + API deps, web app packages, .env + .env.local, Postgres, migrations
```

`make setup` covers steps 4 and 5. The web app is expected next door at
`../creator-agent-web` (override with `make setup WEB=/path/to/frontend`).

Edit `.env`. The only required line is:

```
GEMINI_API_KEY=your-key
```

Never put a real key in `.env.example`: it's committed to git.

Useful settings (all optional; defaults are the recommended values):

| Setting | Default | What it does |
| --- | --- | --- |
| `DEFAULT_STYLE` | `dialogue` | `dialogue` (cast speaks on camera), `mixed` or `narrated` |
| `DEFAULT_LANGUAGE` | `en` | `en` or `ta` |
| `SCRIPT_MODEL` | `gemini-4-carbon` | Script writer. Any Gemini text model, or `anthropic:claude-opus-5-5` |
| `GEMINI_IMAGE_MODEL` | `gemini-3-pro-image` | Frames and character sheets (`gemini-nano-banana-2.1` is about 2.7× cheaper) |
| `OMNI_RESOLUTION` | `1080p` | AI video resolution |
| `VIDEO_PROVIDER` | `omni` | `omni` (AI video), `none` (stills with camera moves), `engine` (legacy) |
| `MUSIC_ENABLED` | `true` | Lyria score under every film |
| `BURN_SUBTITLES` | `false` | Burn speaker-labelled subtitles into the film (adds a small transcription step) |
| `JOB_TTL_SECONDS` | `86400` | How long a film and its files are kept (24h). The Usage page keeps cost history forever. A film still being made is never deleted |
| `MONTHLY_BUDGET_USD` | `300` | App-side monthly cap |
| `BUDGET_MODE` | `block` | `block` refuses new videos past the cap (`degrade` only applies if `REQUIRE_AI_VIDEO=false`) |
| `REQUIRE_AI_VIDEO` | `true` | Every scene must be AI video. If a scene can't get a clip, the film stops and Resume finishes it later; it never ships stills |
| `SPEND_RATE_LIMIT_USD_PER_10MIN` | `9` | Pacing under Google's spend-rate limit (Tier 1: $10 per 10 min; Tier 2: $200) |

## 5. Start the database

Already done by `make setup`. On later runs, `make dev` starts Postgres for you.

```bash
make db-up      # Postgres on :5432, waits until healthy
make migrate    # after pulling new code
make db-shell   # psql, if you want to look inside
```

## 6. Create your channel narrator voice (once)

Narrated and Mixed films use one consistent narrator. Design it from a description:

```bash
make voice GENDER=male VOICE_LANG=en-IN \
  DESCRIBE="A warm, cinematic documentary narrator in his thirties. Deep, smooth baritone, calm and intimate"
```

This writes `NARRATOR_VOICE_ID` to `.env` and saves `narrator_preview.wav`; restart the API to use it.

- Make `GENDER` match the description, and describe the voice's qualities rather than naming a real person (Google blocks voice look-alikes).
- `VOICE_LANG` sets the accent: `en-US`, `en-IN`, `en-GB`, or `ta-IN` for a Tamil narrator.
- There is one narrator setting, so a second run replaces the first voice in `.env`. Note the old id if you want to switch back.
- Skipping this step uses the prebuilt "Kore" voice.

## 7. Check everything

```bash
make doctor        # free
make doctor-live   # one tiny real call per model, ~$0.20
```

It ends with **Ready to generate**, or lists what to fix.

## 8. Run it

```bash
make dev       # API on :8000 + web app on :3000; Ctrl+C stops both
```

| Command | Use it when |
| --- | --- |
| `make api` / `make web` | You want each in its own terminal |
| `make web-prod` | Faster pages: builds the web app and serves it |
| `make api-reload` | You're editing backend code. It restarts on every save, which interrupts a film in progress (Resume continues it) |
| `make stop` / `make status` | Something is still holding port 8000 or 3000 / see what's up |
| `make health` / `make budget` | Model check, Tamil support / month-to-date spend |
| `make open` | Open the web app in your browser |

Open <http://localhost:3000>, write one line, pick a style and language, and select **Make the film**. A 30s film takes about 5–10 minutes.

## Choosing a style

- **Dialogue**: the characters act the story and say their lines on camera, lip-synced, one AI clip per scene. Captions show who's speaking.
- **Mixed**: the narrator sets the story up and wraps it, and the key moments cut to the cast talking.
- **Narrated**: your channel narrator tells the story over AI-filmed scenes.

Tamil films are Tamil end to end: speech, captions and post copy. AI-video speech in Tamil is colloquial, so captions follow what was actually said.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Create page says the monthly budget is reached | Raise `MONTHLY_BUDGET_USD`, or wait until the 1st |
| Doctor: model not visible | Your key's project lacks access or billing. Enable billing in AI Studio |
| Tamil option is rejected | `brew install libraqm`, then restart the API |
| Film stopped with "AI video couldn't be generated" | Omni's daily quota (40 clips/day on Tier 1) is used up. It resets at midnight Pacific; then select Resume, and finished clips are kept. Or select **Finish with N stills** to finish now: those scenes use their still image and no more AI video is charged |
| Log says a scene was "blocked by Google's content rule" | Handled automatically: the scene is rewritten to comply (and its start frame redrawn if needed), and the change is saved in `rescue_NN.json` next to the film. If a scene is still blocked after two rewrites, the film stops: existing heroes, brands or real people in the idea are the usual cause, so describe an original character instead |
| Film is slow on Tier 1 | Spend-rate pacing ($10 per 10 min). Normal until you reach Tier 2 |
| `make dev` says the address is already in use | An old API or web server is still running: `make stop`, then `make dev` |
| Silent video | TTS daily quota (100/day per voice model) is used up; it falls back to the lite model, then silence |
| HTTP 402 from Google | Prepaid credits are at $0; top up in AI Studio |

## Tests

```bash
make test        # API tests, offline (mock providers)
make check       # API tests + web app tests, typecheck, lint and build
make test-live   # real API smoke tests (~$0.50)
make test-cov    # API tests with coverage
```

Paid quality experiments: `make evals` (golden prompts), `make probe-g1` (dialogue and lip-sync),
`make bakeoff-images`, `make bakeoff-scripts`. Results land in `evals/output/`.
