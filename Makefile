.PHONY: weights setup test lint web app dev demo all tunnel tunnel-ngrok screenshots clean

PORT ?= 8000

setup:
	uv sync
	cd web && npm ci
	uv run nbstripout --install || true

# Downloads model weights once into models/weights, then checks each loads with HF_HUB_OFFLINE=1.
weights:
	uv run python -m scripts.weights

test:
	uv run pytest

lint:
	uv run ruff check . --fix && uv run ruff format .

web:
	cd web && npm run build

app: web
	uv run uvicorn api.main:app --host 0.0.0.0 --port $(PORT)

dev:
	( uv run uvicorn api.main:app --reload --port $(PORT) & cd web && npm run dev )

# Stub pipeline: synthetic data -> model -> artifacts -> app. Replace internals tomorrow, keep the targets.
demo:
	uv run python -m src.pipeline --stub
	$(MAKE) app

# Rebuilds every number from raw data. Slides and UI must match this output.
all:
	uv run python -m src.pipeline

tunnel:
	cloudflared tunnel --url http://localhost:$(PORT)

tunnel-ngrok:
	ngrok http $(PORT)

screenshots:
	cd web && npm run screenshots

clean:
	rm -rf web/dist models/*.pkl reports/metrics.json
