# Storyboard: one-line commands for the API, web app, database and checks.
# Run `make` (or `make help`) to list everything.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       := .venv/bin/python
# First Python >= 3.11 found (pyenv included); override with PYTHON=/path/to/python3.12
PYTHON   ?= $(shell for p in python3.13 python3.12 python3.11 $(HOME)/.pyenv/shims/python3 python3; do \
              command -v $$p >/dev/null 2>&1 && $$p -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
              && { command -v $$p; break; }; done)
WEB      ?= ../storyboard_agent_frontend
API_PORT ?= 8000
WEB_PORT ?= 3000
API_URL  := http://localhost:$(API_PORT)
STAMP    := $(shell date +%Y%m%d-%H%M%S)

# Voice Design (make voice DESCRIBE="..." GENDER=male VOICE_LANG=en-IN)
DESCRIBE   ?=
GENDER     ?= female
VOICE_LANG ?= en-US
VOICE_NAME ?= Storyboard narrator

.PHONY: help setup install install-web env db-up db-down db-logs db-shell migrate db-reset \
        dev api api-reload web web-prod stop status health budget open logs \
        doctor doctor-live voice test test-cov test-live test-web typecheck lint build-web check \
        evals evals-mock probe-g1 bakeoff-images bakeoff-scripts diagram clean clean-artifacts

##@ Getting started
help: ## Show this list
	@awk 'BEGIN {FS = ":.*##"} /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
	/^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

setup: install install-web env db-up migrate ## First-time setup: deps, .env files, database
	@echo "Done. Add GEMINI_API_KEY to .env, then run: make doctor && make dev"

install: ## Create .venv and install the API with dev deps
	@test -x $(PY) || { [ -n "$(PYTHON)" ] || { echo "Need Python 3.11+ (brew install python@3.12, or set PYTHON=...)"; exit 1; }; \
	  echo "creating .venv with $(PYTHON)"; $(PYTHON) -m venv .venv; }
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

install-web: ## Install web app packages
	cd $(WEB) && npm install

env: ## Create .env and the web app's .env.local from the examples (never overwrites)
	@test -f .env || { cp .env.example .env && echo "created .env"; }
	@test -f $(WEB)/.env.local || { cp $(WEB)/.env.example $(WEB)/.env.local && echo "created $(WEB)/.env.local"; }

##@ Database
db-up: ## Start Postgres (Docker) and wait until it's healthy
	docker compose up -d --wait postgres

db-down: ## Stop Postgres (keeps data)
	docker compose down

db-logs: ## Tail Postgres logs
	docker compose logs -f postgres

db-shell: ## Open psql
	docker exec -it storyboard_postgres psql -U storyboard storyboard

migrate: ## Apply database migrations
	$(PY) -m alembic upgrade head

db-reset: ## Delete ALL database data and recreate it (CONFIRM=yes)
	@[ "$(CONFIRM)" = "yes" ] || { echo "This deletes every job and the usage history. Re-run with CONFIRM=yes"; exit 1; }
	docker compose down -v
	$(MAKE) db-up migrate

##@ Run
dev: db-up ## Start the API and web app together (Ctrl+C stops both)
	@trap 'kill 0' INT TERM; \
	$(MAKE) --no-print-directory api & \
	$(MAKE) --no-print-directory web & \
	wait

api: ## Start the API on :8000 (use this while making films)
	$(PY) -m uvicorn app.main:app --port $(API_PORT)

api-reload: ## Start the API with auto-reload (code edits interrupt running films)
	$(PY) -m uvicorn app.main:app --port $(API_PORT) --reload

web: ## Start the web app in dev mode on :3000
	cd $(WEB) && npx next dev -p $(WEB_PORT)

web-prod: build-web ## Build and serve the web app (faster pages)
	cd $(WEB) && npx next start -p $(WEB_PORT)

stop: ## Stop anything listening on the API and web ports
	@for p in $(API_PORT) $(WEB_PORT); do \
	  pids=$$(lsof -ti tcp:$$p); \
	  if [ -n "$$pids" ]; then kill -9 $$pids && echo "stopped :$$p"; else echo ":$$p not running"; fi; \
	done

status: ## Show what's running
	@for p in $(API_PORT) $(WEB_PORT); do \
	  if lsof -ti tcp:$$p >/dev/null; then echo ":$$p up"; else echo ":$$p down"; fi; \
	done
	@docker compose ps --format '{{.Name}}: {{.Status}}' 2>/dev/null || true

health: ## API health: models, Tamil support, mock mode
	@out=$$(curl -fsS $(API_URL)/health 2>/dev/null) || { echo "API isn't running on :$(API_PORT). Start it with: make api"; exit 1; }; \
	echo "$$out" | $(PY) -m json.tool

budget: ## This month's spend vs the cap
	@out=$$(curl -fsS $(API_URL)/budget 2>/dev/null) || { echo "API isn't running on :$(API_PORT). Start it with: make api"; exit 1; }; \
	echo "$$out" | $(PY) -m json.tool

open: ## Open the web app in your browser
	open http://localhost:$(WEB_PORT)

##@ Setup checks and voice
doctor: ## Check keys, models, ffmpeg, Tamil fonts, database (free)
	$(PY) -m scripts.doctor

doctor-live: ## Same, plus one tiny real call per model (~$0.20)
	$(PY) -m scripts.doctor --live

voice: ## Design the narrator voice: DESCRIBE="..." GENDER=male VOICE_LANG=en-IN
	@[ -n "$(DESCRIBE)" ] || { echo 'Usage: make voice DESCRIBE="calm documentary narrator..." GENDER=male VOICE_LANG=en-IN'; exit 1; }
	$(PY) -m scripts.design_voice --name "$(VOICE_NAME)" --describe "$(DESCRIBE)" --gender $(GENDER) --language $(VOICE_LANG)
	@echo "Restart the API (make stop && make dev) to use the new voice."

##@ Tests
test: ## API tests (offline, no cost)
	$(PY) -m pytest -q

test-cov: ## API tests with a coverage report
	$(PY) -m pytest -q --cov=app --cov-report=term-missing

test-live: ## Real-API smoke tests (~$0.50)
	$(PY) -m pytest -m live -q

test-web: ## Web app tests
	cd $(WEB) && npm test

typecheck: ## Web app type check
	cd $(WEB) && npm run typecheck

lint: ## Web app lint
	cd $(WEB) && npm run lint

build-web: ## Production build of the web app
	cd $(WEB) && npm run build

check: test test-web typecheck lint build-web ## Everything that must pass before a commit

##@ Quality experiments (paid unless noted)
evals: ## Golden-prompt eval run on real models; LIMIT=2 for fewer
	$(PY) -m evals.run $(if $(LIMIT),--limit $(LIMIT))

evals-mock: ## Golden-prompt eval run offline (free)
	$(PY) -m evals.run --mock $(if $(LIMIT),--limit $(LIMIT))

probe-g1: ## Dialogue/lip-sync probe (~$3-4); OUT=dir
	$(PY) -m scripts.probe_g1 --out $(or $(OUT),evals/output/g1-$(STAMP))

bakeoff-images: ## Image-model bake-off; OUT=dir
	$(PY) -m scripts.bakeoff_images --out $(or $(OUT),evals/output/images-$(STAMP))

bakeoff-scripts: ## Script-model bake-off (Gemini vs Claude); OUT=dir
	$(PY) -m scripts.bakeoff_scripts --out $(or $(OUT),evals/output/scripts-$(STAMP))

diagram: ## Print the pipeline diagram (Mermaid) for the README
	$(PY) -m scripts.render_pipeline_diagram

##@ Clean up
clean: ## Remove caches and the web build
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache .coverage $(WEB)/.next

clean-artifacts: ## Delete every generated film and frame (CONFIRM=yes)
	@[ "$(CONFIRM)" = "yes" ] || { echo "This deletes all films in artifacts_store/. Re-run with CONFIRM=yes"; exit 1; }
	rm -rf artifacts_store/*
