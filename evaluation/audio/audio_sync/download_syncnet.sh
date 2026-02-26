#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/joonson/syncnet_python.git"

# Directory of this script (i.e., audio/audio-sync)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/syncnet_python"

echo "Target directory for syncnet_python: ${TARGET_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed or not in PATH." >&2
  exit 1
fi

if [ -d "${TARGET_DIR}" ]; then
  echo "Error: ${TARGET_DIR} already exists." >&2
  echo "Please remove or rename the existing 'syncnet_python' directory first," >&2
  echo "then rerun this script." >&2
  exit 1
fi

echo "Cloning SyncNet repository from ${REPO_URL} ..."
git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"

# Apply NumPy compatibility fix: np.int was removed in NumPy 1.24+
BOX_UTILS="${TARGET_DIR}/detectors/s3fd/box_utils.py"
if [ -f "${BOX_UTILS}" ]; then
  sed 's/\.astype(np\.int)/.astype(int)/' "${BOX_UTILS}" > "${BOX_UTILS}.tmp" && mv "${BOX_UTILS}.tmp" "${BOX_UTILS}"
  echo "Applied NumPy compatibility fix to detectors/s3fd/box_utils.py"
fi

echo "Done. Local copy of syncnet_python is now available at:"
echo "  ${TARGET_DIR}"

