# Video Evaluation

This directory contains scripts for **video-level evaluation** in two forms:

1. **FD (Fréchet Distance)** — distribution difference between two sets of frame-level (e.g. 3DMM) features.
2. **FVD (Fréchet Video Distance)** — distribution distance between two sets of videos using an I3D feature extractor.

---

## Directory layout

```
video/
├── fd/
│   └── compute_fd_features.py   # Fréchet Distance between two .npy feature sets
├── fvd/
│   ├── preprocess_resize224.py  # Resize MP4 videos to 224×224 (for FVD input)
│   ├── frechet_video_distance.py # FVD computation (I3D features + Fréchet distance)
│   └── run.sh                   # Example script to run FVD with multiple seeds
└── README.md
```

---

## FD: Fréchet Distance between two feature sets

Use **`compute_fd_features.py`** to compute the **Fréchet Distance (FID-style)** between two sets of frame-level feature vectors stored as `.npy` files. Typical use: compare **predicted vs ground-truth 3DMM** (or other per-frame) features.

- **Input**: Two directories, each containing `.npy` files. Each file should have shape `(T, C)` (time steps × feature dimension). The script recursively finds all `.npy` under each directory and stacks them before computing mean and covariance.
- **Output**: A single scalar FID value (lower is better for distribution match).

**Usage**

```bash
cd video/fd
python compute_fd_features.py \
  --pred_dir /path/to/predictions   \
  --gt_dir   /path/to/ground_truth
```

**Dependencies**: `numpy`, `scipy` (no GPU required).

---

## FVD: Frechet Video Distance

FVD measures the distribution distance between **real (GT)** and **generated** videos using I3D features. Recommended workflow:

1. **Resize** both GT and generated videos to **224×224** with `preprocess_resize224.py`.
2. **Run** `frechet_video_distance.py` (e.g. via `run.sh`) on the resized directories.

### Step 1: Resize videos to 224×224

Use **`preprocess_resize224.py`** to batch-resize MP4s to a fixed size (default 224×224). Run it **once for GT** and **once for generated** videos.

- **Note**: Audio is not preserved (video-only output). Codec is typically `mp4v` (depends on your OpenCV build).

**Usage**

```bash
cd video/fvd

# Ground-truth videos
python preprocess_resize224.py \
  --in_dir  /path/to/gt/videos \
  --out_dir /path/to/gt_videos_224 \
  --size 224 --workers 8 --recursive

# Generated videos
python preprocess_resize224.py \
  --in_dir  /path/to/generated/videos \
  --out_dir /path/to/gen_videos_224 \
  --size 224 --workers 8 --recursive
```

Options: `--recursive` to search subdirs, `--workers` for parallelism, `--fps` to force output FPS, `--no_skip_existing` to overwrite existing outputs.

### Step 2: Compute FVD

Use **`frechet_video_distance.py`** to compute FVD between the two **resized** video directories. The script uses a TorchScript I3D model (StyleGAN-V / TFHub-matching), samples clips of fixed length, extracts features, then computes the Fréchet distance between the two feature distributions.

**Usage (single run)**

```bash
cd video/fvd
python frechet_video_distance.py \
  --dir_a /path/to/gt_videos_224 \
  --dir_b /path/to/gen_videos_224 \
  --num_videos 140 \
  --num_frames 256 \
  --batch_size 32 \
  --seed 42 \
  --detector_path /path/to/i3d_torchscript.pt
```

- **`--dir_a`**: Directory of resized videos (e.g. GT).
- **`--dir_b`**: Directory of resized videos (e.g. generated).
- **`--num_videos`**: Number of videos to use from each directory (must have at least this many valid files).
- **`--num_frames`**: Frames per clip fed to I3D.
- **`--detector_path`**: Local path to the I3D TorchScript model (`i3d_torchscript.pt`). If not provided, the script can download it from a default URL (requires network).

**Using `run.sh`**

`run.sh` runs FVD with **multiple seeds** (e.g. 0–4) and reports results per seed. Edit the paths inside `run.sh` to point to your resized directories and detector:

```bash
cd video/fvd
# Edit run.sh: set dir_a, dir_b, num_videos, detector_path, etc.
bash run.sh
```

Example content (customise paths):

```bash
for s in 0 1 2 3 4; do
  python frechet_video_distance.py \
    --dir_a /path/to/gt_videos_224 \
    --dir_b /path/to/gen_videos_224 \
    --num_videos 140 \
    --num_frames 256 \
    --batch_size 32 \
    --seed $((42+s)) \
    --detector_path /path/to/i3d_torchscript.pt
done
```

**Dependencies**: `torch`, `torchvision`, `numpy`, `scipy`, `tqdm`. The I3D model can be downloaded automatically or provided via `--detector_path`.

---

## Summary

| Metric | Script | Input | Output |
|--------|--------|--------|--------|
| **FD** | `fd/compute_fd_features.py` | Two dirs of `.npy` (T×C) | Single FID value |
| **FVD** | `fvd/preprocess_resize224.py` → `fvd/frechet_video_distance.py` (e.g. via `run.sh`) | Two dirs of MP4 → two dirs of 224×224 MP4 | FVD value(s) per run/seed |
