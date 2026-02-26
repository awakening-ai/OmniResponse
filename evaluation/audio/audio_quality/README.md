Audio Quality Evaluation (UTMOSv2)
=================================

This directory contains utilities for **objective audio quality evaluation**
based on the official **UTMOSv2** MOS prediction system  
([UTMOSv2 GitHub repo](https://github.com/sarulab-speech/UTMOSv2.git)).

The goal is to estimate Mean Opinion Score (MOS) for speech/audio clips and to
compute simple aggregated statistics over a folder of `.wav` files.

---

Directory layout
----------------

- `utmosv2/`  
  Local copy of the official UTMOSv2 library (Python package).  
  This folder is obtained directly from the upstream repository using the
  `download_utmosv2.sh` script (see below).  
  It contains all core code for loading the pretrained model, preprocessing
  audio, and predicting MOS.

- `utmosv2_avg.py`  
  Helper script that uses the UTMOSv2 library to:
  - load a pretrained UTMOSv2 model
  - run MOS prediction over a folder of `.wav` files
  - compute the average MOS and optionally print per‑file scores

- `run_avg_utmosv2.sh`  
  Very small convenience wrapper around `utmosv2_avg.py`.  
  Example usage:

  ```bash
  bash run_avg_utmosv2.sh
  # internally this calls:
  # python utmosv2_avg.py --wav-dir your-audio-dir
  ```

- `download_utmosv2.sh`  
  Utility script that **clones the official UTMOSv2 GitHub repository and
  copies only the `utmosv2` subfolder** into this `audio_quality` directory.

---

Environment and dependencies
----------------------------

For a minimal working environment, make sure the following packages are
installed (see the project‑level `requirements.txt` for a consolidated list):

- `torch`, `torchaudio`
- `transformers`
- `timm`
- `librosa`
- `numpy`, `scipy`
- `torchvision`

The exact versions and CUDA builds depend on your local hardware and Python
environment. You can typically start with:

```bash
pip install -r requirements.txt
```

from the project root.

---

Updating / installing the UTMOSv2 code
--------------------------------------

To fetch (or refresh) the local `utmosv2` folder from the official repository,
run the provided script from inside this directory:

```bash
cd audio/audio_quality
bash download_utmosv2.sh
```

What the script does:

1. Clones `https://github.com/sarulab-speech/UTMOSv2.git` into a temporary
   directory.
2. Copies **only** the `utmosv2/` subfolder from the cloned repository into
   `audio/audio_quality/utmosv2`.
3. Deletes the temporary clone.

Notes:

- If `audio/audio_quality/utmosv2` already exists, the script will **refuse to
  overwrite it** and will exit with an error message.  
  You can delete or rename the existing directory manually and then rerun the
  script if you want a fresh copy.
- The script requires:
  - `git` installed and available in `PATH`
  - Internet access to GitHub

---

Running MOS evaluation
----------------------

Assuming you have already:

1. Installed the required Python packages.
2. Downloaded the UTMOSv2 code into `audio/audio_quality/utmosv2`.

You can compute average MOS for a directory of `.wav` files with:

```bash
cd audio/audio_quality
python utmosv2_avg.py --wav-dir /path/to/your/wav_dir
```

or simply use the small wrapper:

```bash
cd audio/audio_quality
bash run_avg_utmosv2.sh
```

