Text Evaluation Toolkit (JSON)
==============================

This folder contains simple scripts for evaluating text generation quality when
your data is stored as one JSON file per example.

Each JSON file is expected to have a top‑level field:

```json
{
  "text": "string with the content to evaluate"
}
```

For **pairwise metrics** (METEOR, ROUGE‑L, BERTScore), you should have:

- one directory with generated hypotheses (`GEN_DIR`)
- one directory with reference texts (`REF_DIR`)

Both directories must contain `.json` files with **matching filenames**.

For **Distinct** diversity metrics, only the generated directory is required.

---

Directory structure
-------------------

- `compute_meteor.py`  
  Compute **METEOR** between generated and reference JSON files.

- `compute_rouge-l.py`  
  Compute **ROUGE‑L** between generated and reference JSON files.

- `compute_bertscore.py`  
  Compute **BERTScore** (P, R, F1) between generated and reference JSON files.

- `compute_distinct.py`  
  Compute **Distinct‑1** and **Distinct‑2** over the generated texts only
  (lexical diversity).

All scripts read `.json` files from the given folder(s), extract the `"text"`
field, filter out invalid/empty samples, and then compute the corresponding
metric.

---

Environment and dependencies
----------------------------

**Python version**

- Python 3.8+ is recommended.

**Core Python packages**

Install these packages before running the text evaluation scripts:

```bash
pip install tqdm nltk numpy bert-score
```

The `bert-score` package will in turn download a transformer model and may
require a backend such as **PyTorch**. If your environment does not already
have it, install a recent PyTorch version compatible with your platform.

**NLTK resources (for METEOR)**

`compute_meteor.py` automatically calls:

- `nltk.download("punkt")`
- `nltk.download("punkt_tab")`
- `nltk.download("wordnet")`
- `nltk.download("omw-1.4")`

If your environment does not allow downloads at runtime, you can pre‑download
these resources manually:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

One‑shot evaluation script
--------------------------

Use `run_all_text_metrics.sh` to compute all text metrics for a pair of
directories in one command.

**Usage**

```bash
bash run_all_text_metrics.sh GEN_DIR REF_DIR
```

- `GEN_DIR`: directory containing generated `.json` files.
- `REF_DIR`: directory containing reference `.json` files.

Both must have matching filenames.

**Main metrics (to maximize, ↑):**

- METEOR ↑
- BERTScoreF1 ↑
- ROUGE‑L ↑
- Distinct‑2 ↑

The script prints two compact, copy‑friendly lines:

- **Main line** – the key metrics you usually care about:

  ```text
  MAIN  METEOR=0.3123  BERTScoreF1=0.8456  ROUGE-L=0.4021  Distinct-2=0.2734
  ```

- **Extra line** – additional metrics for reference:

  ```text
  EXTRA Distinct-1=0.5123  BERTScoreP=0.8523  BERTScoreR=0.8390  NumPairs=100
  ```

You can easily copy these two lines into a spreadsheet or log file.

---

Individual scripts
------------------

You can also run each metric script separately.

**METEOR**

```bash
python compute_meteor.py \
  --gen_dir /path/to/gen_json \
  --ref_dir /path/to/ref_json
```

**ROUGE‑L**

```bash
python compute_rouge-l.py \
  --gen_dir /path/to/gen_json \
  --ref_dir /path/to/ref_json
```

**BERTScore (P/R/F1)**

```bash
python compute_bertscore.py \
  --gen_dir /path/to/gen_json \
  --ref_dir /path/to/ref_json \
  --lang en
```

**Distinct‑1 / Distinct‑2**

```bash
python compute_distinct.py \
  --gen_dir /path/to/gen_json
```

