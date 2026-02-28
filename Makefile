.PHONY: install install-dev test lint clean run-pipeline run-stage download-models

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/

lint-fix:
	ruff check --fix src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/

download-models:
	python scripts/download_models.py

run-pipeline:
	python scripts/run_pipeline.py --config configs/default.yaml

run-stage:
	@echo "Usage: make run-stage STAGE=0 CONFIG=configs/default.yaml"
	python scripts/run_stage.py --config $(CONFIG) --stage $(STAGE)

smoke-test:
	python scripts/run_pipeline.py --config configs/default.yaml --smoke-test
