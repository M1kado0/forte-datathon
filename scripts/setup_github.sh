#!/usr/bin/env bash
# P1, once: create the repo, push, protect main. Needs gh >= 2.94 and `gh auth login`.
# Teammates are invited by hand afterwards (link printed at the end).
# Usage: scripts/setup_github.sh [repo-name]
set -euo pipefail
NAME=${1:-forte-datathon}

gh --version | head -1
git init -b main 2>/dev/null || true
git add -A && git commit -m "chore: datathon template" || true
gh repo create "$NAME" --private --source . --push

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# Squash-only merges, auto-delete branches
gh api -X PATCH "repos/$REPO" -F allow_squash_merge=true -F allow_merge_commit=false \
  -F allow_rebase_merge=false -F delete_branch_on_merge=true >/dev/null

# Protect main: PR required, no force-push. (Private repos need GitHub Pro/Team for protection;
# if this fails, make the repo public after the event or just follow the rule by convention.)
gh api -X PUT "repos/$REPO/branches/main/protection" --input - <<'JSON' && echo "main protected" || echo "⚠️ protection failed — see comment"
{ "required_status_checks": null, "enforce_admins": false,
  "required_pull_request_reviews": { "required_approving_review_count": 0 },
  "restrictions": null, "allow_force_pushes": false, "allow_deletions": false }
JSON

# Labels used by tickets
for l in mvp0 mvp1 stretch build hypothesis; do gh label create "$l" --force >/dev/null; done
echo "Repo: https://github.com/$REPO"
echo "Invite teammates: https://github.com/$REPO/settings/access"
