#!/usr/bin/env bash
set -euo pipefail
root=runs/pose-robustness-v2/fruit-completion-v1
while [[ ! -s "$root/fruit_language-base/completed.json" ]]; do
  pid=$(cat "$root/language-base-job-pid.txt")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Reference language evaluation ended without completion.' >&2
    exit 1
  fi
  sleep 10
done
cp code/scripts/run-fruit-ppo-language-after-reference-v1.sh "$root/sources/scripts/"
exec bash code/scripts/run-fruit-language-overlap-v1.sh ppo
