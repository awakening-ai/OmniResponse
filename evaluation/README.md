# Evaluation Toolkit

This repository provides scripts and pipelines for evaluating **text**, **audio**, and **video** outputs (e.g. from TTS, lip-sync, or video generation models). Each modality has its own subdirectory with dedicated READMEs and scripts.

---

## Directory layout

```
evaluation/
├── README.md                 # this file
├── requirements.txt         # Python dependencies (text + audio + video)
│
├── text/                    # Text generation metrics
│   ├── README.md
│   ├── compute_meteor.py
│   ├── compute_rouge-l.py
│   ├── compute_bertscore.py
│   ├── compute_distinct.py
│   └── run_all_text_metrics.sh   # one-shot: all metrics for gen_dir vs ref_dir
│
├── audio/
│   ├── audio_quality/       # MOS prediction (UTMOSv2)
│   │   ├── README.md
│   │   ├── download_utmosv2.sh
│   │   ├── utmosv2_avg.py
│   │   └── run_avg_utmosv2.sh
│   │
│   └── audio_sync/          # Lip-sync scores (SyncNet, LSE-D)
│       ├── README.md
│       ├── combine_av.py
│       ├── download_syncnet.sh
│       ├── calculate_scores_real_videos.sh
│       ├── calculate_scores_real_videos.py
│       └── compute_avg_lse_d.py
│
└── video/                   # Video distribution metrics
    ├── README.md
    ├── fd/                  # Fréchet Distance between feature sets (e.g. 3DMM)
    │   └── compute_fd_features.py
    └── fvd/                 # Frechet Video Distance (I3D)
        ├── preprocess_resize224.py
        ├── frechet_video_distance.py
        └── run.sh
```

---

## Quick start

### 1. Environment

From the repository root:

```bash
pip install -r requirements.txt
```

You may need to install **PyTorch** (and optionally CUDA) separately. For **ffmpeg** (used by audio_sync and some video/audio steps), install via your system or module (e.g. `module load ffmpeg`).

### 2. Text metrics

- **Input**: Two directories of JSON files with a top-level `"text"` field; same filenames in both.
- **Main metrics**: METEOR ↑, BERTScore-F1 ↑, ROUGE-L ↑, Distinct-2 ↑

```bash
cd text
bash run_all_text_metrics.sh /path/to/generated_json_dir /path/to/reference_json_dir
```

See [text/README.md](text/README.md) for per-metric scripts and dependencies (e.g. NLTK, bert-score).

### 3. Audio quality (MOS)

- **Input**: A directory of `.wav` files.
- **Output**: Average MOS from UTMOSv2.

First fetch the UTMOSv2 code (only the `utmosv2` subfolder), then run:

```bash
cd audio/audio_quality
bash download_utmosv2.sh
python utmosv2_avg.py --wav-dir /path/to/wav_dir
```

See [audio/audio_quality/README.md](audio/audio_quality/README.md).

### 4. Audio–video sync (LSE-D)

- **Input**: Combined A+V videos (e.g. from `combine_av.py`: video + audio → MP4).
- **Output**: Per-file LSE-D and **avg. LSE-D**.

Typical workflow:

```bash
cd audio/audio_sync
python combine_av.py -v /path/to/video -a /path/to/audio -o /path/to/combined_av
bash download_syncnet.sh
cd syncnet_python && bash download_model.sh && cd ..
module load ffmpeg   # or install ffmpeg
bash calculate_scores_real_videos.sh /path/to/combined_av
```

See [audio/audio_sync/README.md](audio/audio_sync/README.md).

### 5. Video metrics (FD & FVD)

- **FD**: Fréchet distance between two sets of `.npy` frame-level features (e.g. 3DMM).
- **FVD**: Frechet Video Distance; resize videos to 224×224, then run the I3D-based script.

```bash
# FD
cd video/fd
python compute_fd_features.py --pred_dir /path/to/pred_npy --gt_dir /path/to/gt_npy

# FVD
cd video/fvd
python preprocess_resize224.py --in_dir /path/to/gt_videos --out_dir /path/to/gt_224 --size 224 --recursive
python preprocess_resize224.py --in_dir /path/to/gen_videos --out_dir /path/to/gen_224 --size 224 --recursive
bash run.sh   # edit paths inside run.sh first
```

See [video/README.md](video/README.md).

---

## Summary

| Modality   | Metric / output   | Location           | One-line usage |
|-----------|-------------------|--------------------|----------------|
| Text      | METEOR, BERTScore, ROUGE-L, Distinct | `text/`           | `run_all_text_metrics.sh GEN_DIR REF_DIR` |
| Audio     | MOS (UTMOSv2)     | `audio/audio_quality/` | `download_utmosv2.sh` then `utmosv2_avg.py --wav-dir DIR` |
| Audio+Video | LSE-D (SyncNet) | `audio/audio_sync/`   | `combine_av.py` → `download_syncnet.sh` → `download_model.sh` → `calculate_scores_real_videos.sh DIR` |
| Video     | FD (features)     | `video/fd/`       | `compute_fd_features.py --pred_dir P --gt_dir G` |
| Video     | FVD (I3D)        | `video/fvd/`      | Resize with `preprocess_resize224.py`, then `run.sh` |

For details, dependencies, and options, see the README in each subdirectory.
