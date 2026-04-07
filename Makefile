.PHONY: lint format type

all:
	make format
	make lint
	make type

lint:
	ruff check src/

format:
	ruff format src/

type:
	ty check
