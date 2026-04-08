.PHONY: lint format type build-visualizer run-visualizer

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

test-activation-speed:
	uv run src/scripts/compare_activation_speed.py datasets/data.csv

build-visualizer:
	cd visualizer/frontend && npm install && npm run build

run-visualizer:
	uv run mlp-visualizer --static visualizer/frontend/dist