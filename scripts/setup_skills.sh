#!/usr/bin/env bash
# Run ONCE tonight by P1 in the repo root, then commit everything. Other laptops just `git pull`.
# ⚠️ Don't re-run or update skills during the event.
set -euo pipefail

# 1) Harness-neutral skills dir: .agents/skills is the source, .claude/skills points to it
mkdir -p .agents/skills
[ -e .claude/skills ] || { mkdir -p .claude && ln -s ../.agents/skills .claude/skills; }

# 2) Matt Pocock skills (grill-with-docs, to-spec, to-tickets, code-review, handoff)
#    ⚠️ verify skill names with `npx skills add mattpocock/skills --list` first
npx skills add mattpocock/skills \
  --skill grill-with-docs --skill to-spec --skill to-tickets --skill code-review --skill handoff \
  -a claude-code -a codex

# 3) Design: Impeccable + shadcn + web-design-guidelines (do NOT add frontend-design / ui-ux-pro-max)
npx impeccable install --providers=claude,codex --scope=project
npx skills add shadcn/ui -a claude-code -a codex
npx skills add vercel-labs/agent-skills --skill web-design-guidelines -a claude-code -a codex

# 4) MCP servers (project scope for Claude; Codex users run the codex lines on their own laptop)
(cd web && npx shadcn@latest mcp init --client claude)
claude mcp add --scope project playwright npx @playwright/mcp@latest
# Codex: npx shadcn@latest mcp init --client codex ; codex mcp add playwright npx "@playwright/mcp@latest"
# Codex: open /hooks and approve Impeccable's hooks.

echo "Done. Now: claude → /context (check AGENTS.md loads) → type / and confirm the skills are listed."
echo "Then: /impeccable init (paste docs/IMPECCABLE_BRAND.md) → /impeccable document → /impeccable hooks on"
