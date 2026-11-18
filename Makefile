.PHONY: install data graph test lint eval api mcp docker

RAW ?= data/raw/DBahn-berlin
WEEKS ?= 250902_250909

install:
	pip install -e ".[dev,ollama,openai,anthropic]"

# Fetch the DBahn-Berlin dataset (~190 MB, all weeks) with a sparse git checkout
data:
	@test -d data/raw/datasets || git clone --depth 1 --filter=blob:none --sparse https://github.com/damslab/datasets data/raw/datasets
	cd data/raw/datasets && git sparse-checkout set DBahn-berlin
	ln -sfn datasets/DBahn-berlin $(RAW)

graph:
	transit-graph build --raw $(RAW) --weeks $(WEEKS)

test:
	pytest -q

lint:
	ruff check src tests evals && ruff format --check src tests evals

eval:
	python evals/run_evals.py --provider $${PROVIDER:-offline} $${MODEL:+--model $$MODEL}

api:
	uvicorn transit_graph.api:app --reload

mcp:
	transit-graph-mcp

docker:
	docker compose up --build
