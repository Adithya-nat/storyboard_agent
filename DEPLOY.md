# Deploying Creator Agent

This guide puts Creator Agent online for other people: a homepage, sign-in, and a studio where
each person makes films with their own Gemini key. Everything here uses free tiers and
open-source tools. To run it only for yourself on your own machine, use [SETUP.md](SETUP.md).

Each step ends with a **Check**. Don't move on until it passes.

## 1. How the pieces fit

```
  GitHub Pages                     your server (one VM)                       Neon (free Postgres)
  ┌──────────────┐  "Start     ┌─────────────────────────────────────┐      ┌──────────────────────┐
  │  homepage    │──creating"─▶│ Caddy (HTTPS)                        │      │ schema creator_auth: │
  │  site/       │             │  ├─ app.<domain> → web  (Next.js)    │─────▶│   users, sessions    │
  └──────────────┘             │  │    sign-in (Better Auth)          │      │ schema public:       │
                               │  └─ api.<domain> → api  (FastAPI)    │─────▶│   jobs, logs, costs  │
                               │       films + ffmpeg, files on disk  │      └──────────────────────┘
                               └─────────────────────────────────────┘
                                         │ each user's own key
                                         ▼
                                   Google Gemini API
```

| Piece | Where | Cost | Limits that matter |
| --- | --- | --- | --- |
| Homepage | GitHub Pages | Free for public repos | None for a static page |
| Database | Neon Free | Free, no card | 100 compute-hours/month, 1 GB storage, 5 GB egress. Over the limit it **pauses until next month**; it never bills you. |
| Server | Oracle Cloud Always Free (recommended) | Free | 4 Arm cores, 24 GB RAM, 200 GB disk |
| Sign-in | Better Auth inside the web app | Free, MIT | None |
| AI | Each user's own Gemini key | Paid by the user | Google's quotas per key |

Why one server and not "serverless": a film takes 10 to 20 minutes of background work with
ffmpeg and local files, and the studio streams progress live. Free serverless plans sleep,
cap request time or throttle background CPU, and films die halfway.

Videos never go into the database. They live on the server's disk for 24 hours, so Neon's
storage and egress limits are far away.

## 2. The homepage (GitHub Pages)

The workflow `.github/workflows/pages.yml` publishes `site/` on every push to `main` that
touches it.

1. In the `creator-agent` repo: **Settings → Pages → Build and deployment → Source: GitHub
   Actions.** (Already done for the main repo.)
2. **Actions → Homepage → Run workflow** to publish now.
3. Once the studio is online (step 6), set where "Start creating" goes:
   **Settings → Secrets and variables → Actions → Variables → New repository variable**,
   name `APP_URL`, value `https://app.<your-domain>`. Then run the workflow again.
   Until `APP_URL` exists, the buttons say "Run it yourself" and link to the README.

Optional custom domain: Settings → Pages → Custom domain, then add a `CNAME` record pointing at
`<your-user>.github.io`.

**Check:** <https://adithya-nat.github.io/creator-agent/> loads, and after step 3 "Start
creating" opens your sign-in page.

## 3. The database (Neon)

1. Sign up at <https://neon.com> (no card). Create a project in the region closest to your
   server.
2. **Connect → turn "Connection pooling" off** and copy the connection string. The host must
   **not** contain `-pooler`. The API keeps a few long-lived connections and needs a direct
   connection for its startup setup; the pooled endpoint breaks both.
3. Paste it as is. `postgresql://…?sslmode=require&channel_binding=require` is fine; the API
   converts it for its driver.
4. Create the tables. The API's tables are created automatically when the API container
   starts. Create the sign-in tables once, from a computer with Node 22:

   ```bash
   cd creator-agent-web && npm ci
   DATABASE_URL='<neon url>' BETTER_AUTH_URL=https://app.<domain> \
   BETTER_AUTH_SECRET='<from step 4>' npm run auth:migrate
   ```

**Check:** in the Neon console's Tables view, schema `creator_auth` has `user`, `session`,
`account`, `verification` and `jwks`.

Keeping inside the free tier: the database sleeps after 5 idle minutes. The API only wakes it
when someone uses the studio or a film expires. Watch **Usage → Compute** in the console; a
quiet month uses a small fraction of the 100 hours.

## 4. Sign-in (GitHub and Google)

Generate the two secrets first:

```bash
openssl rand -base64 32   # BETTER_AUTH_SECRET
openssl rand -hex 32      # ASSET_SIGNING_SECRET
```

**GitHub:** <https://github.com/settings/developers> → OAuth Apps → New OAuth App.
- Homepage URL: `https://app.<domain>`
- Authorization callback URL: `https://app.<domain>/api/auth/callback/github`
- Copy the Client ID, generate a client secret.
- A GitHub OAuth app has one callback URL, so make a second app for local testing with
  `http://localhost:3000/api/auth/callback/github`.

**Google:** <https://console.cloud.google.com/apis/credentials>
1. Configure the OAuth consent screen: External, app name "Creator Agent", your email. Scopes:
   `email`, `profile`, `openid`.
2. Create credentials → OAuth client ID → Web application.
   - Authorized JavaScript origin: `https://app.<domain>`
   - Authorized redirect URI: `https://app.<domain>/api/auth/callback/google`
3. While the consent screen is in "Testing", only the test users you list can sign in. Publish
   it when you're ready for anyone. Email and profile scopes don't need Google's review.

**The beta allowlist:** anyone can sign in, but only people in `ALLOWED_USERS` can make films;
everyone else sees "You're on the list". Use emails, `@company.com` for a whole domain, or `*`
for everyone. Change it and restart the API (`docker compose … up -d api`).

## 5. The server

### Option A (recommended): Oracle Cloud Always Free + Docker Compose

1. Create an account at <https://www.oracle.com/cloud/free/> (a card is checked for identity;
   Always Free resources aren't charged).
2. **Create a VM instance:**
   - Image: Ubuntu 24.04
   - Shape: Ampere `VM.Standard.A1.Flex`, 4 OCPUs and 24 GB memory (all of the free
     allowance)
   - Boot volume: 100 GB
   - Add your SSH public key
   - If it says "out of capacity", try another availability domain or try again later. This
     is common.
3. **Open ports 80 and 443** in two places:
   - Networking → your VCN → Security Lists → Default → Add Ingress Rules: source
     `0.0.0.0/0`, TCP, ports `80,443`.
   - On the VM itself (Oracle's Ubuntu image blocks them in iptables):

     ```bash
     sudo iptables -I INPUT 6 -p tcp -m multiport --dports 80,443 -m state --state NEW -j ACCEPT
     sudo netfilter-persistent save
     ```

4. **Domains.** Point two names at the VM's public IP:
   - **Your own domain:** A records `app` and `api`.
   - **Free:** a [DuckDNS](https://www.duckdns.org) name such as `creatoragent.duckdns.org`.
     With only one name, use `app.` for the web app and a second DuckDNS name for the API.
   - **Quick test:** `app.<ip-with-dashes>.sslip.io` and `api.<ip-with-dashes>.sslip.io` need
     no sign-up. Certificates can occasionally fail because many people share that domain.
5. **Install Docker and get the code:**

   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER && newgrp docker
   git clone https://github.com/Adithya-nat/creator-agent
   git clone https://github.com/Adithya-nat/creator-agent-web
   cd creator-agent
   cp deploy/.env.example deploy/.env && nano deploy/.env   # fill in every value
   docker compose -f deploy/compose.prod.yml --env-file deploy/.env up -d --build
   ```

   The first build takes about 10 minutes on the Arm VM. Caddy gets HTTPS certificates on
   its own.

**Check:**
- `curl https://api.<domain>/health` shows `"auth_mode":"jwt","key_mode":"byok"`.
- `https://app.<domain>` redirects to the sign-in page.
- Sign in with an allowlisted account, add a Gemini key (**Test and save** should list the
  models), and make a 30-second film.

### Option B: Coolify (a web UI instead of the command line)

[Coolify](https://coolify.io) is an open-source, self-hosted alternative to Heroku/Vercel.
Install it on the same kind of VM:

```bash
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | sudo bash
```

Then open `http://<ip>:8000`.
- Add both GitHub repos as **Dockerfile** applications. The `Dockerfile` in each repo root is
  ready.
- Copy the environment from `deploy/compose.prod.yml`.
- Give the API a persistent volume at `/data`.
- Assign the `app.` and `api.` domains.

Coolify runs its own proxy for HTTPS, so don't also start the Caddy service.

### Other options

- **Any small VPS** (Hetzner, DigitalOcean and similar, about $5/month): Option A works
  unchanged. Use 2+ cores and 4 GB+ RAM; ffmpeg is the heavy part.
- **A spare computer at home** with [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
  (`cloudflared` is open source): no open ports needed. Point the tunnel's two hostnames at
  `localhost:3000` and `localhost:8000`, and skip Caddy.
- **Web app on Vercel Hobby** (free, not open source, non-commercial only): set the same env
  vars and keep the API on your server.

## 6. Connect the homepage

Set the `APP_URL` repository variable (step 2.3) and re-run the Homepage workflow.

**Check:** the homepage's "Start creating" and "Sign in" go to `https://app.<domain>/login`.

## 7. Environment reference

API (`creator-agent`). In hosted mode, `deploy/compose.prod.yml` sets these from `deploy/.env`.

| Variable | Local default | Hosted | What it does |
| --- | --- | --- | --- |
| `AUTH_MODE` | `none` | `jwt` | `jwt` requires a sign-in token on every request |
| `AUTH_JWKS_URL` | | `https://app.<domain>/api/auth/jwks` | Where the API gets the web app's public signing keys |
| `AUTH_ISSUER` | | `https://app.<domain>` | Must equal the web app's `BETTER_AUTH_URL` |
| `AUTH_AUDIENCE` | `creator-agent-api` | same | Must match the web app's `AUTH_AUDIENCE` |
| `ALLOWED_USERS` | | `you@x.com,@team.com` | Beta allowlist; `*` = everyone |
| `KEY_MODE` | `server` | `byok` | `byok`: each user's key, never stored |
| `GEMINI_API_KEY` | your key | empty | Server key, only used in `server` mode |
| `ASSET_SIGNING_SECRET` | random per start | required | Signs video and live-stream links |
| `DATABASE_URL` | local Docker | Neon direct URL | |
| `CORS_ORIGINS` | `http://localhost:3000` | `https://app.<domain>` | |
| `USER_MONTHLY_BUDGET_USD` | 50 | 50 | Per-user cap on estimated spend |
| `MAX_ACTIVE_JOBS_PER_USER` / `MAX_ACTIVE_JOBS` | 1 / 3 | same | Films rendering at once (hosted only) |
| `JOB_TTL_SECONDS` | 86400 | 86400 (maximum) | Everything is deleted this long after a film starts |
| `ARTIFACT_ROOT` | `./artifacts_store` | `/data/artifacts` | Where films live until they're deleted |

Web app (`creator-agent-web`):

| Variable | Local | Hosted | What it does |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | `https://api.<domain>` | Build time |
| `NEXT_PUBLIC_AUTH_ENABLED` | `false` | `true` | Build time; turns on sign-in |
| `NEXT_PUBLIC_HOMEPAGE_URL` | Pages URL | same | The logo link on the sign-in page |
| `BETTER_AUTH_URL` | `http://localhost:3000` | `https://app.<domain>` | |
| `BETTER_AUTH_SECRET` | | required | Signs sessions |
| `DATABASE_URL` | | Neon direct URL | Same database as the API |
| `GITHUB_CLIENT_ID/SECRET`, `GOOGLE_CLIENT_ID/SECRET` | | required | OAuth apps from step 4 |
| `AUTH_EMAIL_PASSWORD` | `false` | `false` | Local testing only: email + password sign-up |

## 8. Running it

- **Update:**

  ```bash
  cd creator-agent && git pull
  cd ../creator-agent-web && git pull
  cd ../creator-agent
  docker compose -f deploy/compose.prod.yml --env-file deploy/.env up -d --build
  ```

  Migrations run on start. Restarting the API interrupts films that are rendering; users can
  press Resume.
- **Logs:** `docker compose -f deploy/compose.prod.yml logs -f api`. Keys are redacted from logs.
- **Disk:** films are deleted 24 hours after they start, automatically. `docker system df`
  shows what's used.
- **Add testers:** edit `ALLOWED_USERS` in `deploy/.env`, then `… up -d api`.
- **Rotate secrets:** changing `BETTER_AUTH_SECRET` signs everyone out. Changing
  `ASSET_SIGNING_SECRET` only breaks open video links until the page reloads.
- **Backups:** Neon Free can restore to any point in the last 6 hours. Films aren't backed up;
  they're deleted after 24 hours by design.

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| "redirect_uri mismatch" on sign-in | The OAuth callback must be exactly `https://app.<domain>/api/auth/callback/github` (or `/google`) |
| Signed in, but every API call is 401 | `AUTH_ISSUER` must equal `BETTER_AUTH_URL`, and the API must reach `AUTH_JWKS_URL` (try `curl` from the server) |
| "You're on the list" for your own account | Add your email to `ALLOWED_USERS` and restart the API |
| API won't start: SSL or "prepared statement" errors | Use Neon's direct connection string, not the `-pooler` one |
| "Neon compute suspended" / SQLSTATE 53000 | The month's free compute is used up. Upgrade or wait for the 1st |
| Key test says AI video isn't available | Enable billing on the Google Cloud project that owns the key |
| Tamil films refused (422) | The API image includes libraqm; if you run outside Docker, `brew install libraqm` / `apt install libraqm0` |
| Homepage 404 right after setup | Pages needs one successful workflow run: Actions → Homepage → Run workflow |
| Certificates fail with sslip.io | Use your own domain or DuckDNS |
