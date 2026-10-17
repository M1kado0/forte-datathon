# Serving: public address for the jury (#30, TZ-F.4)

Serving machine: Ubuntu, RTX 5080 (P2's box). Online from 11:00 until results.

## Stage 1: quick tunnel (cloudflared)
```bash
# one-time: install cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
make app                     # API + web on :8000 (needs API_KEY in .env)
make tunnel                  # prints https://<random>.trycloudflare.com; the URL changes on every run
curl https://<random>.trycloudflare.com/api/health
make jury-sim PUBLIC_URL=https://<random>.trycloudflare.com
```
`scripts/jury_sim.py` checks health, that a missing key gives 401, then runs 5 parallel jobs (p95 < 240 s,
every answer validated with `contracts.schemas.JobResponse`) and 8 parallel ~145 s videos (built locally
from our clips with ffmpeg, all within 300 s). Exit code 1 on any failure. `--stress 0` skips the stress phase.

Quick tunnels have no SSE and no uptime guarantee: the job API polls, so this is fine.

## Stage 2: by 21:00 (not yet applied)
- Tailscale Funnel on :8000, systemd units for the API and the tunnel/Funnel, sleep and unattended-upgrades off.
- To be written and applied on the serving machine with sudo; reboot test: the address must answer with no manual steps.
