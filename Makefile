.PHONY: dev stop build test lint migrate seed clean pull-models

# ── Development ───────────────────────────────
dev:
	docker compose up --build -d

stop:
	docker compose down

build:
	docker compose build --no-cache

logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-celery:
	docker compose logs -f celery-worker

# ── Database ──────────────────────────────────
migrate:
	docker compose exec backend alembic upgrade head

migrate-create:
	docker compose exec backend alembic revision --autogenerate -m "$(msg)"

# ── Testing ───────────────────────────────────
test:
	docker compose exec backend pytest -v

test-cov:
	docker compose exec backend pytest --cov=app --cov-report=html

# ── Linting ───────────────────────────────────
lint:
	docker compose exec backend ruff check app/
	docker compose exec backend ruff format --check app/

lint-fix:
	docker compose exec backend ruff check --fix app/
	docker compose exec backend ruff format app/

# ── Ollama Models ─────────────────────────────
pull-models:
	docker compose exec ollama ollama pull llama3.1:8b
	docker compose exec ollama ollama pull nomic-embed-text

# ── Seed Data ─────────────────────────────────
seed:
	docker compose exec backend python -m app.scripts.seed

# ── Cleanup ───────────────────────────────────
clean:
	docker compose down -v --remove-orphans
	docker system prune -f
