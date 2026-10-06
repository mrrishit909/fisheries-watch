export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55370/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8370

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m fw.seed --if-empty
dev:            ## API with reload on :8370, jobs run by a local worker
	($(PY) -m core.jobs fw.api &) && $(PY) -m uvicorn fw.api:app --reload --port 8370
test:
	$(PY) -m pytest --cov=fw --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) coast-viewer-demo 32 10 "GET /v1/events/suspicious?hours=48" "GET /v1/vessels/ae03fbdf-c338-5861-9510-6d169de322d8/timeline?hours=60" "GET /v1/areas/ade93e57-65fb-5f98-ab7c-e3e9b3b698e1/activity?hours=48" 
evaluate:       ## held-out fabs -> docs/evaluation.md
	$(PY) -m fw.evaluate > docs/evaluation.md
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test evaluate reset
