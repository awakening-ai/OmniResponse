# Run from audio_sync/ (parent of syncnet_python/).
# Requires: (1) syncnet_python/download_model.sh already run from syncnet_python/
#           (2) ffmpeg in PATH
# Usage: bash calculate_scores_real_videos.sh /path/to/video/dir

set -e
VIDEO_DIR="$1"
DATA_DIR_NAME="${2:-tmp_dir}"

if [ -z "$VIDEO_DIR" ] || [ ! -d "$VIDEO_DIR" ]; then
  echo "Usage: $0 <video_directory>" >&2
  echo "  e.g. $0 /path/to/listener/cb" >&2
  exit 1
fi

# Absolute paths so they work after cd into syncnet_python
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR_ABS="${BASE_DIR}/${DATA_DIR_NAME}"

if [ ! -f "${BASE_DIR}/syncnet_python/data/syncnet_v2.model" ]; then
  echo "Error: SyncNet model not found. Run from syncnet_python first:" >&2
  echo "  cd ${BASE_DIR}/syncnet_python && bash download_model.sh && cd -" >&2
  exit 1
fi

if [ ! -f "${BASE_DIR}/syncnet_python/detectors/s3fd/weights/sfd_face.pth" ]; then
  echo "Error: S3FD weights not found. Run from syncnet_python first:" >&2
  echo "  cd ${BASE_DIR}/syncnet_python && bash download_model.sh && cd -" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found in PATH. Install ffmpeg or load a module (e.g. module load ffmpeg)." >&2
  exit 1
fi

rm -f all_scores.txt
for eachfile in "$VIDEO_DIR"/*; do
  [ -f "$eachfile" ] || continue
  videofile_abs="$(cd "$(dirname "$eachfile")" && pwd)/$(basename "$eachfile")"
  (cd "${BASE_DIR}/syncnet_python" && python run_pipeline.py --videofile "$videofile_abs" --reference wav2lip --data_dir "$DATA_DIR_ABS")
  python calculate_scores_real_videos.py --videofile "$eachfile" --reference wav2lip --data_dir "$DATA_DIR_ABS" --initial_model "${BASE_DIR}/syncnet_python/data/syncnet_v2.model" >> all_scores.txt
done
echo "Scores appended to all_scores.txt"

# Compute and print average of first column (LSE-D) from all_scores.txt
if [ -s all_scores.txt ]; then
  avg_lse_d=$(awk '{ sum += $1; n++ } END { if (n > 0) printf "%.6f", sum/n; else print "nan" }' all_scores.txt)
  echo "avg. LSE-D: ${avg_lse_d}"
else
  echo "avg. LSE-D: (no scores in all_scores.txt)"
fi
