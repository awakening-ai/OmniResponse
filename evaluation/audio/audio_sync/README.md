# Audio-Visual Sync (SyncNet) — Lip-Sync Evaluation

This directory provides a full pipeline to **combine audio and video** and then compute **lip-sync scores (LSE-D)** using the [SyncNet](https://github.com/joonson/syncnet_python) model. No files inside `syncnet_python/` are modified; only scripts in this parent folder are used and documented here.

---

## Directory layout

| File / folder | Description |
|---------------|-------------|
| `combine_av.py` | Merges same-named MP4 (video) and WAV (audio) files into one MP4 per pair. |
| `download_syncnet.sh` | Clones the `syncnet_python` repo into this directory (run once). |
| `syncnet_python/` | Upstream SyncNet code. After cloning, run `download_model.sh` inside it. |
| `calculate_scores_real_videos.sh` | Runs the SyncNet pipeline on a folder of MP4s and writes per-file scores to `all_scores.txt`, then prints **avg. LSE-D**. |
| `calculate_scores_real_videos.py` | Loads SyncNet and scores cropped face tracks (called by the shell script). |
| `compute_avg_lse_d.py` | Standalone script: reads `all_scores.txt` and prints the average of the first column (LSE-D). |

---

## Step-by-step workflow

Run these steps from the **`audio_sync`** directory (or adjust paths if you run from elsewhere).

### Step 1 — Combine audio and video into MP4

Use `combine_av.py` to merge **video** (e.g. `video/*.mp4`) and **audio** (e.g. `audio/*.wav`) by matching filenames (same stem). Output is written to a folder of your choice.

```bash
cd audio_sync

python combine_av.py \
  --video_dir /path/to/video/folder    \
  --audio_dir /path/to/audio/folder    \
  --output_dir /path/to/combined_av_videos
```

- **`--video_dir`**: directory containing `.mp4` files (top-level only).
- **`--audio_dir`**: directory containing `.wav` files; names must match video stems (e.g. `name.mp4` ↔ `name.wav`).
- **`--output_dir`**: directory where merged `.mp4` files will be written.

Requires **ffmpeg** in `PATH`.

---

### Step 2 — Clone SyncNet repository

If you do not yet have the `syncnet_python` folder, run:

```bash
cd audio_sync
bash download_syncnet.sh
```

This clones `https://github.com/joonson/syncnet_python` into `audio_sync/syncnet_python`. The script will not overwrite an existing `syncnet_python/` directory.

---

### Step 3 — Download SyncNet and S3FD weights

From inside `syncnet_python`, run the official download script, then go back to `audio_sync`:

```bash
cd audio_sync/syncnet_python
bash download_model.sh
cd ..
```

This creates:

- `syncnet_python/data/syncnet_v2.model`
- `syncnet_python/detectors/s3fd/weights/sfd_face.pth`

---

### Step 4 — Load ffmpeg and run lip-sync scoring

Ensure **ffmpeg** is available (e.g. on a cluster, load the module). Then run the scoring script on the **folder of combined MP4s** produced in Step 1:

```bash
cd audio_sync

module load ffmpeg   # or however ffmpeg is provided on your system

bash calculate_scores_real_videos.sh /path/to/combined_av_videos
```

- **First argument**: directory containing the `.mp4` files to score.
- **Optional second argument**: working directory for pipeline intermediates (default: `tmp_dir`).

Output:

- Per-file scores are appended to **`all_scores.txt`** (one line per video: first column = LSE-D, second = confidence).
- At the end, the script prints **avg. LSE-D** (average of the first column).

---

### Step 5 — Average LSE-D from `all_scores.txt`

`all_scores.txt` has two columns per line, for example:

```
3.8251367 0.09926796
3.89664 0.06507015
5.0368156 0.022602081
...
```

- **First column**: LSE-D (offset distance) per video.
- **Second column**: confidence.

**Option A — From the shell script**  
`calculate_scores_real_videos.sh` already computes and prints the average of the first column at the end of the run, e.g.:

```text
avg. LSE-D: 4.123456
```

**Option B — Standalone script**  
To recompute the average from an existing `all_scores.txt`:

```bash
cd audio_sync
python compute_avg_lse_d.py
# By default reads ./all_scores.txt; optional: python compute_avg_lse_d.py /path/to/all_scores.txt
```

Output is a single line: **avg. LSE-D: &lt;value&gt;** (and optionally the number of samples).

---

## Summary

| Step | Command / action |
|------|-------------------|
| 1 | `python combine_av.py -v <video_dir> -a <audio_dir> -o <output_dir>` |
| 2 | `bash download_syncnet.sh` |
| 3 | `cd syncnet_python && bash download_model.sh && cd ..` |
| 4 | `module load ffmpeg` then `bash calculate_scores_real_videos.sh /path/to/combined_av_videos` |
| 5 | Read **avg. LSE-D** from the script output or run `python compute_avg_lse_d.py` on `all_scores.txt` |

---

## Dependencies

- **Python**: packages from `syncnet_python/requirements.txt` (e.g. PyTorch, OpenCV, SciPy). Install with `pip install -r syncnet_python/requirements.txt` after Step 2.
- **ffmpeg**: required for `combine_av.py` and the SyncNet pipeline. Install or load a module (e.g. `module load ffmpeg`) before Step 1 and Step 4.
