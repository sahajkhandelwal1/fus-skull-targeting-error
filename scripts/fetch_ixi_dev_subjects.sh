#!/usr/bin/env bash
# Downloads a small (~5 subject) batch of raw IXI T1 MRI scans for pipeline dev,
# per Phase 1 / Immediate Next Action #3.
#
# The official source (biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/)
# was returning 403 Forbidden for the IXI subfolder specifically (confirmed
# broken from multiple independent networks, not just an IP block) as of
# 2026-07-23. Using the Kaggle mirror "kbacon/ixi-t1" instead, which hosts the
# same raw (non-skull-stripped) per-subject NIfTI files under their original
# IXI naming convention.
#
# Requires: `pip install kaggle` and Kaggle API auth configured
# (~/.kaggle/access_token or `kaggle auth login`).

set -euo pipefail

OUT_DIR="$(dirname "$0")/../data/raw/ixi_t1_dev"
mkdir -p "$OUT_DIR"

FILES=(
  "IXI002-Guys-0828-T1.nii/IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01.nii"
  "IXI012-HH-1211-T1.nii/IXI012-HH-1211-3DBRAINIXMADisoTFE12_-s3T111_-0301-00003-000001-01.nii"
  "IXI013-HH-1212-T1.nii/IXI013-HH-1212-IXIMADisoTFE12_-s3T111_-0301-00003-000001-01.nii"
  "IXI016-Guys-0697-T1.nii/IXI016-Guys-0697-IXI3DMPRAG_-s231_-0301-00003-000001-01.nii"
  "IXI021-Guys-0703-T1.nii/IXI021-Guys-0703-IXI3DMPRAG_-s232_-0301-00003-000001-01.nii"
)

for f in "${FILES[@]}"; do
  echo "=== $f ==="
  kaggle datasets download kbacon/ixi-t1 -f "$f" -p "$OUT_DIR" --unzip
done

# kaggle CLI names the zip after the .nii file (e.g. foo.nii.zip) but doesn't
# always auto-unzip it; clean up any leftover archives.
for z in "$OUT_DIR"/*.zip; do
  [ -e "$z" ] || continue
  unzip -o "$z" -d "$OUT_DIR"
  rm "$z"
done

echo "Downloaded ${#FILES[@]} subjects to $OUT_DIR"
