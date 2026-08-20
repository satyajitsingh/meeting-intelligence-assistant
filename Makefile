# Meeting Intelligence Assistant — common tasks.
# Run `make help` for the list.

BACKEND  := backend
FRONTEND := frontend
PY       := $(BACKEND)/.venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup backend-setup frontend-setup dev-backend dev-frontend \
        test test-backend test-frontend check check-backend check-frontend \
        lint typecheck build up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: backend-setup frontend-setup ## Install backend and frontend dependencies

backend-setup: ## Create the backend venv and install dependencies
	python3 -m venv $(BACKEND)/.venv
	$(PY) -m pip install --upgrade pip
	cd $(BACKEND) && .venv/bin/python -m pip install -e ".[dev]"
	@test -f $(BACKEND)/.env || cp $(BACKEND)/.env.example $(BACKEND)/.env
	@echo "Add provider keys to $(BACKEND)/.env when you want answers or transcription."

frontend-setup: ## Install frontend dependencies
	cd $(FRONTEND) && npm install
	@test -f $(FRONTEND)/.env.local || cp $(FRONTEND)/.env.example $(FRONTEND)/.env.local

dev-backend: ## Run the API on http://localhost:8000
	cd $(BACKEND) && .venv/bin/python -m uvicorn app.main:app --reload --port 8000

dev-frontend: ## Run the UI on http://localhost:3000
	cd $(FRONTEND) && npm run dev

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests (excludes network-dependent ones)
	cd $(BACKEND) && .venv/bin/python -m pytest

test-frontend: ## Run frontend tests
	cd $(FRONTEND) && npx vitest run

check: check-backend check-frontend ## Run every lint, type and test check

check-backend: ## Backend: pytest, ruff, mypy
	cd $(BACKEND) && .venv/bin/python -m pytest
	cd $(BACKEND) && .venv/bin/python -m ruff check . --exclude .venv
	cd $(BACKEND) && .venv/bin/python -m ruff format --check . --exclude .venv
	cd $(BACKEND) && .venv/bin/python -m mypy app

check-frontend: ## Frontend: eslint, tsc, build, vitest
	cd $(FRONTEND) && npm run lint
	cd $(FRONTEND) && npm run typecheck
	cd $(FRONTEND) && npm run build
	cd $(FRONTEND) && npx vitest run

up: ## Start both services with Docker
	docker compose up --build

down: ## Stop Docker services
	docker compose down

clean: ## Remove caches and build output
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache
	rm -rf $(FRONTEND)/.next $(FRONTEND)/tsconfig.tsbuildinfo
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
