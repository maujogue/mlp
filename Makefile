.PHONY: lint format type

all:
	make format
	make lint
	make type

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

type:
	uv run ty check
