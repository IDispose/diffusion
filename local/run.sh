#!/usr/bin/env bash
# Run RFdiffusion in the local env built by ./local/setup.sh. All arguments are passed
# straight through to run_inference.py as hydra overrides, e.g.
#
#   ./local/run.sh inference.output_prefix=outputs/binder \
#                  inference.input_pdb=target.pdb \
#                  'contigmap.contigs=[A160-242/A302-504 129-129]' \
#                  'ppi.hotspot_res=[A200,A204]' diffuser.T=50
#
# Note diffuser.T must be >= 15 (RFdiffusion asserts this).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x rfdiff-env/bin/python ] || { echo "run ./local/setup.sh first" >&2; exit 1; }
export DGLBACKEND=pytorch
exec rfdiff-env/bin/python RFdiffusion/run_inference.py "$@"
