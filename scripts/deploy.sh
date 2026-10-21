#!/bin/bash
# Deploy to Liara with the current git commit recorded in the image (the bot logs it at start).
# Usage: bash scripts/deploy.sh   (commit first: the recorded commit must match the uploaded code)
set -e; cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain bot agent)" ] || { echo "uncommitted changes in bot/ or agent/ - commit first"; exit 1; }
git rev-parse --short HEAD > bot/GIT_COMMIT
liara deploy --app wallex-agent --team-id 6a86b17a129b5d10d46e4dae --port 8000 --detach
