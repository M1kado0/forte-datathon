#!/usr/bin/env bash
# Stage 2 of #30: API under systemd, Tailscale Funnel on :8000, no sleep, no automatic updates.
# Run as your normal user on the serving machine (it calls sudo itself):  bash scripts/serve_install.sh
# Check afterwards, and again after a reboot:                             bash scripts/serve_install.sh check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
RUN_USER="$(id -un)"

check() {
  local bad=0
  for unit in forte-api forte-funnel tailscaled; do
    printf '%-18s enabled=%s active=%s\n' "$unit" "$(systemctl is-enabled "$unit" 2>&1)" "$(systemctl is-active "$unit" 2>&1)"
  done
  for unit in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    [ "$(systemctl is-enabled "$unit" 2>&1)" = masked ] || { echo "NOT masked: $unit"; bad=1; }
  done
  for unit in unattended-upgrades apt-daily.timer apt-daily-upgrade.timer; do
    [ "$(systemctl is-enabled "$unit" 2>&1)" = disabled ] || { echo "NOT disabled: $unit"; bad=1; }
  done
  curl -fsS -m 5 "http://127.0.0.1:$PORT/api/health" && echo || { echo "local /api/health failed"; bad=1; }
  tailscale funnel status || bad=1
  local url
  url="https://$(tailscale status --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
  curl -fsS -m 15 "$url/api/health" && echo || { echo "public $url/api/health failed"; bad=1; }
  echo "PUBLIC_URL=$url"
  return $bad
}

if [ "${1:-}" = check ]; then check; exit $?; fi

command -v tailscale >/dev/null || { echo "tailscale is not installed"; exit 1; }
[ -x "$ROOT/.venv/bin/uvicorn" ] || { echo "run 'make setup' first"; exit 1; }

# 0. Funnel must be allowed for the tailnet (admin console, one click), otherwise the command waits forever
funnel_out="$(timeout 20 tailscale funnel --bg "$PORT" 2>&1 || true)"
if grep -q "not enabled" <<<"$funnel_out"; then
  echo "$funnel_out"; echo "Open the link above, enable Funnel, then rerun this script."; exit 1
fi

# 1. API key and web build (user level, no sudo)
if [ ! -f "$ROOT/.env" ]; then
  echo "API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" > "$ROOT/.env"
  chmod 600 "$ROOT/.env"
  echo "created .env with a new API_KEY (give it to the jury): $(cat "$ROOT/.env")"
fi
if [ ! -f "$ROOT/web/dist/index.html" ]; then
  [ -d "$ROOT/web/node_modules" ] || (cd "$ROOT/web" && npm ci)
  (cd "$ROOT/web" && npm run build)
fi

# 2. systemd units. Funnel config itself survives reboots inside tailscaled; the unit re-applies it as a safeguard.
sudo tee /etc/systemd/system/forte-api.service >/dev/null <<UNIT
[Unit]
Description=Forte datathon API (uvicorn on 127.0.0.1:$PORT)
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$ROOT
Environment=HF_HUB_OFFLINE=1
ExecStart=$ROOT/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port $PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo tee /etc/systemd/system/forte-funnel.service >/dev/null <<UNIT
[Unit]
Description=Tailscale Funnel to the Forte API
After=tailscaled.service forte-api.service network-online.target
Wants=tailscaled.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/tailscale funnel --bg $PORT
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now tailscaled forte-api forte-funnel

# 3. No sleep, no automatic updates (serving box must stay up from 11:00 until results)
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
sudo tee /etc/apt/apt.conf.d/20auto-upgrades >/dev/null <<'CONF'
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
CONF
sudo tailscale set --auto-update=false || true
command -v snap >/dev/null && sudo snap refresh --hold || true

echo; check
