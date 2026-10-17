# Serving: public address for the jury (#30, TZ-F.4)

Serving machine: Ubuntu, RTX 5080 (P2's box). Online from 11:00 until results.
The address the jury gets is the Tailscale Funnel URL (stage 2). ngrok (stage 1) is a quick fallback.

Needs `ffmpeg` on the machine that runs `make jury-sim` (`sudo apt install ffmpeg`), see `docs/SETUP.md`.

## Stage 1: ngrok tunnel (fallback)
One-time: create an ngrok account and run `ngrok config add-authtoken <token>` (token from the ngrok dashboard).
```bash
make app                     # API + web on :8000 (needs API_KEY in .env)
make tunnel                  # ngrok; public URL in the terminal and at http://localhost:4040
curl https://<name>.ngrok-free.dev/api/health
make jury-sim PUBLIC_URL=https://<name>.ngrok-free.dev
```
`scripts/jury_sim.py` checks health (JSON with `status: ok`), that a missing key gives 401, then runs 5 parallel
jobs (p95 < 240 s, every answer validated with `contracts.schemas.JobResponse`; with 5 jobs p95 is the maximum)
and 8 parallel ~145 s videos (built locally from our clips with ffmpeg): the whole batch within 300 s and
every clip within 45 s (`docs/PLAN.md` T10). Exit code 1 on any failure. `--stress 0` skips the stress phase.
Its output is the source of the numbers; paste it into the ticket. Runs against the stub analyzer never decode
video and prove only the plumbing (tunnel, key, schema); the limits count only with the real analyzer.

### Why not cloudflared
On 2026-10-17 `cloudflared` 2026.10.1 never connected on two different networks: QUIC timed out and the TLS
handshake to the edge on port 7844 was reset (also with `--protocol http2`; Cloudflare answers 1033/530).
`make tunnel-cloudflared` stays as a fallback, e.g. from a phone hotspot.

## Stage 2: Tailscale Funnel + systemd (by 21:00)
Tailscale is installed and logged in on the serving machine, and Funnel is enabled on the tailnet (a one-time
approval by the tailnet owner via the link that `tailscale funnel --bg 8000` prints, if it is not).
The URL is stable: `https://<machine>.<tailnet>.ts.net`; `bash scripts/serve_install.sh check` prints it as `PUBLIC_URL=`.

On the serving machine (needs sudo and your password):
```bash
bash scripts/serve_install.sh          # creates .env with an API_KEY if missing, builds web/dist, installs units
bash scripts/serve_install.sh check    # statuses + public /api/health; rerun after a reboot
make jury-sim PUBLIC_URL=<the PUBLIC_URL printed by check>
```
If the script creates `.env`, it does not print the key: read `API_KEY` from `.env` and give it to the jury.

What the script sets up:
- `forte-api.service`: uvicorn on `127.0.0.1:8000` from `.venv`, `Restart=always`, `HF_HUB_OFFLINE=1`.
- `forte-funnel.service`: `tailscale funnel --bg 8000` after `tailscaled` (the Funnel config also persists inside tailscaled).
- Sleep: `sleep/suspend/hibernate/hybrid-sleep.target` masked (GNOME idle suspend on AC was already `nothing`).
- Updates: `unattended-upgrades`, `apt-daily*.timer` disabled, `20auto-upgrades` zeroed, tailscale auto-update off,
  `snap refresh --hold`.

Reboot test (acceptance): reboot, wait about a minute, run `bash scripts/serve_install.sh check` without starting anything by hand.
Give the jury the Funnel URL instead of the ngrok address.
