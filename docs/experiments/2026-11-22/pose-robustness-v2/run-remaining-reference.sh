#!/usr/bin/env bash
set -euo pipefail
# Continue the fixed reference suite after its 20 ordinary fruit cases finish.
# Final outcome files stay unopened during development-only model selection.
bash .local/run-final-reference.sh insert_flower official ordinary
bash .local/run-final-reference.sh take_out_cool_drink official ordinary
for mode in official paraphrase alternate_target; do
  bash .local/run-final-reference.sh select_fruit "$mode" fruit_language
done
