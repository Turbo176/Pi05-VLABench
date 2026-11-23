#!/usr/bin/env bash
set -euo pipefail
source .local/reference-env.sh
gate=runs/pose-robustness-v2/imitation-gate-canonical-lr1e4-300-flower42/gate.json
while ! "$py" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(0 if p.exists() and json.loads(p.read_text())["complete"] else 1)' "$gate"; do
 if ! pgrep -f 'run-imitation-flower-canonical-lr1e4-evaluation.sh|verify_pose_imitation_gate.py .*imitation-gate-canonical-lr1e4-300-flower42$' >/dev/null; then
  echo 'Five-control comparison ended without a completed gate.' >&2; exit 1
 fi
 sleep 5
done
if "$py" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$gate"; then
 echo 'Five-control gate passed; the conditional execution-profile diagnostic is unnecessary.'
 exit 0
fi
exec bash code/scripts/run-imitation-flower-execute1.sh
