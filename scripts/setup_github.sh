#!/usr/bin/env bash
# Setup GitHub : crée le repo, active Pages, push le code.
# Usage: bash scripts/setup_github.sh <github-user> <repo-name>

set -euo pipefail

USER="${1:-}"
REPO="${2:-polymarket-bot}"

if [ -z "$USER" ]; then
  echo "Usage: bash scripts/setup_github.sh <github-user> [repo-name]"
  exit 1
fi

echo "[setup] init git..."
git init -b main 2>/dev/null || true
git add .
git commit -m "feat: initial polymarket bot with ML ensemble + dashboard" || true

echo "[setup] création repo GitHub $USER/$REPO..."
gh repo create "$USER/$REPO" --public --source=. --remote=origin --push

echo "[setup] activation GitHub Pages (Actions source)..."
gh api -X POST "repos/$USER/$REPO/pages" \
  -f "source[branch]=main" -f "source[path]=/" \
  -H "Accept: application/vnd.github+json" 2>/dev/null || \
  echo "  (Pages déjà actif ou permission needed dans Settings → Pages)"

echo "[setup] permissions Actions (write)..."
gh api -X PUT "repos/$USER/$REPO/actions/permissions/workflow" \
  -F default_workflow_permissions='write' \
  -F can_approve_pull_request_reviews='true' 2>/dev/null || true

echo ""
echo "[setup] ✓ terminé !"
echo "  Repo    : https://github.com/$USER/$REPO"
echo "  Actions : https://github.com/$USER/$REPO/actions"
echo "  Pages   : https://$USER.github.io/$REPO/ (dispo sous ~2 min)"
