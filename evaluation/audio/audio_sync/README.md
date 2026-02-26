Audio-Visual Synchronisation (SyncNet)
=====================================

This directory contains utilities related to **audio-visual synchronisation**
based on the **SyncNet** model from the repository  
[`joonson/syncnet_python`](https://github.com/joonson/syncnet_python).

The SyncNet network can be used for tasks such as:

- removing temporal lags between audio and visual streams in a video
- determining which face is speaking in multi-speaker videos

The original code and detailed documentation are provided in the upstream
repository.

---

Directory layout
----------------

- `syncnet_python/`  
  Local clone of the official `syncnet_python` repository.  
  This folder is created by running `download_syncnet.sh` (see below). It
  contains:

  - core model code: `SyncNetModel.py`, `SyncNetInstance.py`
  - demo / pipeline scripts: `demo_syncnet.py`, `run_pipeline.py`,
    `run_syncnet.py`, `run_visualise.py`
  - face detectors and utilities under `detectors/`
  - example data, scripts, and configuration files

- `download_syncnet.sh`  
  Helper script to clone the `syncnet_python` repository into this directory.

You can treat `syncnet_python/` as a standalone project and follow its own
`README.md` for advanced usage.

---

Environment and dependencies
----------------------------

SyncNet relies on:

- **Python packages** listed in `syncnet_python/requirements.txt` (e.g. PyTorch,
  NumPy, SciPy, OpenCV, etc.).
- **System tools**:
  - `ffmpeg` is required for handling audio/video.

A typical setup flow is:

```bash
# 1) Download the SyncNet code
cd audio/audio-sync
bash download_syncnet.sh

# 2) Install Python dependencies from the cloned repo
cd syncnet_python
pip install -r requirements.txt
```

Make sure `ffmpeg` is installed and accessible in your `PATH`. On macOS, for
example:

```bash
brew install ffmpeg
```

---

Fetching / refreshing SyncNet
-----------------------------

To obtain (or refresh) the local copy of `syncnet_python/`, run:

```bash
cd audio/audio-sync
bash download_syncnet.sh
```

The script will:

1. Clone `https://github.com/joonson/syncnet_python` into
   `audio/audio-sync/syncnet_python`.
2. Refuse to overwrite an existing `syncnet_python/` directory (you must delete
   or rename it first if you want a clean clone).

---

Example usage
-------------

Once the repository is cloned and dependencies are installed, you can use the
original demo and pipeline scripts from the upstream project. For example, from
inside `audio/audio-sync/syncnet_python`:

```bash
python demo_syncnet.py \
  --videofile data/example.avi \
  --tmp_dir /path/to/temp
```

For the full pipeline (face detection, cropping, sync scoring, and
visualisation), follow the instructions in the upstream README  
[`syncnet_python`](https://github.com/joonson/syncnet_python).

