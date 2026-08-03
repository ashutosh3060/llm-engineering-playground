.PHONY: install dev test lint fmt typecheck run api bench clean

install:
	python -m pip install -e ".[dev]"

dev: install
	cp -n .env.example .env || true
	@echo "Edit .env, then run: make run"

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy src

run:
	playground ui

api:
	playground serve

bench:
	playground bench datasets/sentiment-classification.yaml -r 5

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
