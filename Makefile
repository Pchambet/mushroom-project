.PHONY: help setup data run report all app test lint format requirements clean

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create the environment from uv.lock
	uv sync --locked

data:  ## Download (cached, checksum-verified) and tidy the UCI file
	uv run mushroom data

run:  ## Run every experiment and redraw the README figures (about 3 min)
	uv run mushroom run

report:  ## Build the static report in site/index.html
	uv run mushroom report

all: data run report  ## Full pipeline

app:  ## Launch the Streamlit dashboard
	uv run streamlit run app.py

test:  ## Unit tests (offline, a few seconds)
	uv run pytest -q

lint:  ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply ruff formatting and safe fixes
	uv run ruff check --fix .
	uv run ruff format .

requirements:  ## Re-export the pinned requirements.txt used by Streamlit Cloud
	uv export --no-hashes --no-dev --no-emit-project --format requirements-txt -o requirements.txt

clean:  ## Remove caches and the downloaded raw file
	rm -rf data/raw .pytest_cache .ruff_cache
