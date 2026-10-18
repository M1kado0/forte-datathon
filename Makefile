.PHONY: weights weights-check setup test lint web app dev demo all tunnel tunnel-cloudflared jury-sim viewpoint screenshots clean

PORT ?= 8000
# Serving uses real weights. For a standalone UI demo: make app FORTE_ANALYZER=stub.
export FORTE_ANALYZER ?= real

setup:
	uv sync
	cd web && npm ci
	uv run nbstripout --install || true

# Downloads model weights once into models/weights, then checks each loads with HF_HUB_OFFLINE=1.
weights:
	uv run python -m scripts.weights

weights-check:
	HF_HUB_OFFLINE=1 uv run python -m scripts.weights --check

test:
	uv run pytest

lint:
	uv run ruff check . --fix && uv run ruff format .

web:
	cd web && npm run build

app: web
	uv run uvicorn api.main:app --host 0.0.0.0 --port $(PORT)

dev: export FORTE_ANALYZER = stub
dev:
	( uv run uvicorn api.main:app --reload --port $(PORT) & cd web && npm run dev )

# Stub pipeline: synthetic data -> model -> artifacts -> app. Replace internals tomorrow, keep the targets.
demo: export FORTE_ANALYZER = stub
demo:
	uv run python -m src.pipeline --stub
	$(MAKE) app

# Rebuilds every number from raw data. Slides and UI must match this output.
all:
	uv run python -m src.rules.types_doc
	uv run python -m src.pipeline
	uv run python -m src.data.insights

# Public URL for the jury via ngrok (outbound 443; cloudflared is blocked on our networks). Start `make app` first.
# The URL is printed in the ngrok UI and at http://localhost:4040.
tunnel:
	ngrok http $(PORT)

# Fallback quick tunnel; needs outbound 7844, which our networks reset. URL changes every run.
tunnel-cloudflared:
	cloudflared tunnel --no-autoupdate --url http://localhost:$(PORT)

# Warped training videos and our own recordings (#43). Embeds about 25 min of video per variant, so it is not in `make all`.
viewpoint:
	uv run python -m scripts.viewpoint_eval

# Jury simulation against a running service: make jury-sim PUBLIC_URL=https://...
jury-sim:
	uv run python -m scripts.jury_sim --url $(PUBLIC_URL)

screenshots:
	cd web && npm run screenshots

clean:
	rm -rf web/dist models/*.pkl reports/metrics.json
