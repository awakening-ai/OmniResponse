#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/sarulab-speech/UTMOSv2.git"

# Directory of this script (i.e., audio/audio_quality)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/utmosv2"

echo "Target directory for utmosv2: ${TARGET_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed or not in PATH." >&2
  exit 1
fi

if [ -d "${TARGET_DIR}" ]; then
  echo "Error: ${TARGET_DIR} already exists." >&2
  echo "Please remove or rename the existing 'utmosv2' directory first," >&2
  echo "then rerun this script." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
echo "Cloning UTMOSv2 into temporary directory: ${TMP_DIR}"

git clone --depth 1 "${REPO_URL}" "${TMP_DIR}/UTMOSv2"

if [ ! -d "${TMP_DIR}/UTMOSv2/utmosv2" ]; then
  echo "Error: 'utmosv2' subfolder not found in cloned repository." >&2
  rm -rf "${TMP_DIR}"
  exit 1
fi

echo "Copying 'utmosv2' subfolder to ${SCRIPT_DIR}"
mv "${TMP_DIR}/UTMOSv2/utmosv2" "${SCRIPT_DIR}/"

rm -rf "${TMP_DIR}"

echo "Done. Local copy of UTMOSv2 is now available at:"
echo "  ${SCRIPT_DIR}/utmosv2"

