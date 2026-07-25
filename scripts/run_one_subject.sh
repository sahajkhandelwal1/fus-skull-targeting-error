#!/usr/bin/env bash
# Runs the full one-subject pipeline (Phases A-G) end to end for a single
# IXI subject already present in data/raw/ixi_t1_dev/. Crosses both Python
# environments (k-Wave on 3.13, mr-to-pct on 3.10) since they can't share
# one venv -- see README.md "Environment setup".
#
# Usage: scripts/run_one_subject.sh IXI002

set -euo pipefail

SUBJECT_ID="${1:?Usage: $0 <subject_id>, e.g. IXI002}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export SSL_CERT_FILE="$(.venv/bin/python3 -m certifi 2>/dev/null || .venv-mrtopct/bin/python3 -m certifi)"

echo "=== Phase A: reorient + resample ==="
source .venv/bin/activate
python3 scripts/prepare_ixi_subject.py "$SUBJECT_ID"
deactivate

echo "=== Phase B: MRI -> pseudo-CT ==="
source .venv-mrtopct/bin/activate
python3 scripts/convert_ixi_subject_to_pct.py "$SUBJECT_ID"

echo "=== Atlas-based VIM target registration ==="
python3 scripts/find_atlas_target.py "$SUBJECT_ID"
deactivate

source .venv/bin/activate

echo "=== Phase C: pseudo-CT -> acoustic maps ==="
python3 scripts/compute_acoustic_maps.py "$SUBJECT_ID"

echo "=== Phase D: through-skull k-Wave simulation ==="
python3 scripts/run_skull_simulation.py "$SUBJECT_ID"

echo "=== Phase E: targeting error / energy loss labels ==="
python3 scripts/compute_targeting_error.py "$SUBJECT_ID"

echo "=== Phase F: skull descriptors ==="
python3 scripts/extract_skull_descriptors.py "$SUBJECT_ID"

echo "=== Phase G: assemble record + summary figure ==="
python3 scripts/assemble_subject_record.py "$SUBJECT_ID"
python3 scripts/make_pipeline_summary_figure.py "$SUBJECT_ID"

echo "=== Done: $SUBJECT_ID ==="
