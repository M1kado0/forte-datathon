# Setup для участников

Цель: `make setup && make test` проходит на каждом ноутбуке. Около 15 минут.

## 1. Нужно всем

| Инструмент | Версия | Зачем | macOS | Windows |
|---|---|---|---|---|
| git | любая свежая | репо, worktree | `brew install git` | `winget install Git.Git` (ставит и Git Bash) |
| gh | ≥ 2.94 | issues, PR, sub-issues для `/to-tickets` | `brew install gh` или `brew upgrade gh` | `winget install GitHub.cli` |
| uv | свежий | Python и все Python-зависимости | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Node.js + npm | ≥ 20 | сборка `web/`, `npx` для MCP | `brew install node` | `winget install OpenJS.NodeJS.LTS` |
| make | любая | цели `make setup/test/app` | уже есть | WSL, или `winget install ezwinports.make` в Git Bash |
| Claude Code и/или Codex CLI | свежий | агенты и скиллы | `npm i -g @anthropic-ai/claude-code` / `npm i -g @openai/codex` | то же |
| playwright-cli | свежий | агент открывает сайты и UI через браузер (вместо Playwright MCP) | `npm i -g @playwright/cli` | то же |

Скилл `playwright-cli` уже лежит в репо (`.agents/skills/playwright-cli`), ставить его не нужно.
Playwright MCP не используем: в `.mcp.json` его нет.

Python отдельно ставить не нужно: uv скачает его сам.

## 2. Только для своей роли

| Инструмент | Кому | Зачем | Установка |
|---|---|---|---|
| cloudflared | тот, кто сервит (P1 или владелец RTX) | публичный URL для жюри (`make tunnel`) | `brew install cloudflared` / `winget install Cloudflare.cloudflared` |
| NVIDIA driver с поддержкой CUDA 12.8 | владелец RTX 5070 (P2) | torch cu128 на GPU | GeForce Experience или nvidia.com/drivers. CUDA Toolkit ставить не нужно |
| Playwright Chromium | P1, P4 | скриншоты UI 1440/390 (`make screenshots`) | `cd web && npx playwright install chromium` |
| ffmpeg | P3 (по желанию) | `ffprobe` для кодека и fps клипов | `brew install ffmpeg` / `winget install Gyan.FFmpeg` |

Не нужно: pandoc (бриф уже лежит в `docs/BRIEF.md`), ngrok (запасной вариант туннеля).

## 3. После установки

```bash
gh auth login
git clone git@github.com:M1kado0/forte-datathon.git && cd forte-datathon
make setup          # uv sync + npm ci в web/
make test           # должно быть "10 passed"
```

Без make (Windows без WSL):
```bash
uv sync
cd web && npm ci && cd ..
uv run pytest
```

## 4. Данные (не в git)

Датасет весит 295 MB и в репо не попадает. Положите его так:
```
data/raw/open_labels.xlsx
data/raw/videos/*.mov      # 42 клипа
```

## Проверка

```bash
git --version; gh --version; uv --version; node --version; make --version; claude --version; playwright-cli --version
```
