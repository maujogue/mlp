.PHONY: lint format type

lint:
	ruff check src/

format:
	ruff format src/

type:
	ty check

all: lint format type