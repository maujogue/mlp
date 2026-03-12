.PHONY: lint format type

lint:
	ruff check src/

format:
	ruff format src/

type:
	ty check

all:
	make format
	make lint
	make type
	