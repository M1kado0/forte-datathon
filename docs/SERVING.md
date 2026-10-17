# Serving: public address for the jury (#30, TZ-F.4)

Serving machine: Ubuntu, RTX 5080 (P2's box). Online from 11:00 until results.

## Stage 1: ngrok tunnel
```bash
make app                     # API + web on :8000 (needs API_KEY in .env)
make tunnel                  # ngrok; public URL in the terminal and at http://localhost:4040
curl https://<name>.ngrok-free.dev/api/health
make jury-sim PUBLIC_URL=https://<name>.ngrok-free.dev
```
`scripts/jury_sim.py` checks health, that a missing key gives 401, then runs 5 parallel jobs (p95 < 240 s,
every answer validated with `contracts.schemas.JobResponse`) and 8 parallel ~145 s videos (built locally
from our clips with ffmpeg, all within 300 s). Exit code 1 on any failure. `--stress 0` skips the stress phase.
Measured on the stub through ngrok: p95 9.5 s, stress 37.0 s, 0 schema errors. The stub never decodes video,
so rerun with the real analyzer.

### Why not cloudflared
On 2026-10-17 `cloudflared` 2026.10.1 never connected on two different networks: QUIC timed out and the TLS
handshake to the edge on port 7844 was reset (also with `--protocol http2`; Cloudflare answers 1033/530).
`make tunnel-cloudflared` stays as a fallback, e.g. from a phone hotspot. Binary: `~/.local/bin/cloudflared`.

## Stage 2: Tailscale Funnel + systemd (by 21:00)
Tailscale is installed and logged in on the serving machine (tailnet name `yeskendir-x870-gaming-x-wifi7.tail8f157f.ts.net`).
Funnel is **not enabled on the tailnet yet**: the owner must open the link that `tailscale funnel --bg 8000` prints
(`https://login.tailscale.com/f/funnel?node=...`) and approve it once.

Then, on the serving machine (needs sudo and your password):
```bash
bash scripts/serve_install.sh          # creates .env with an API_KEY if missing, builds web/dist, installs units
bash scripts/serve_install.sh check    # statuses + public /api/health; rerun after a reboot
make jury-sim PUBLIC_URL=https://yeskendir-x870-gaming-x-wifi7.tail8f157f.ts.net
```
What the script sets up:
- `forte-api.service`: uvicorn on `127.0.0.1:8000` from `.venv`, `Restart=always`, `HF_HUB_OFFLINE=1`.
- `forte-funnel.service`: `tailscale funnel --bg 8000` after `tailscaled` (the Funnel config also persists inside tailscaled).
- Sleep: `sleep/suspend/hibernate/hybrid-sleep.target` masked (GNOME idle suspend on AC was already `nothing`).
- Updates: `unattended-upgrades`, `apt-daily*.timer` disabled, `20auto-upgrades` zeroed, tailscale auto-update off,
  `snap refresh --hold`.

Reboot test (acceptance): reboot, wait about a minute, run `bash scripts/serve_install.sh check` without starting anything by hand.
The public URL is stable, so give the jury this one instead of the ngrok address.
