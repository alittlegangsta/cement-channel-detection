.PHONY: test test-unit test-integration test-smoke lint check-env format

test:
	python -m pytest tests

test-smoke:
	python -m pytest tests/smoke

test-unit:
	python -m pytest tests/unit

test-integration:
	python -m pytest tests/integration

lint:
	python -m ruff check src scripts tests

format:
	python -m ruff format src scripts tests

check-env:
	python scripts/00_check_env.py

fix:
	python -m ruff check src scripts tests --fix
	python -m ruff format src scripts tests