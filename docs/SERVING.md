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

## Stage 2: by 21:00 (not yet applied)
- Tailscale Funnel on :8000, systemd units for the API and the tunnel/Funnel, sleep and unattended-upgrades off.
- To be written and applied on the serving machine with sudo; reboot test: the address must answer with no manual steps.
