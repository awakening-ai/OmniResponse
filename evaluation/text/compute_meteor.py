#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute METEOR for a directory of generated JSON files against reference JSON files.

Assumptions:
- gen_dir and ref_dir both contain .json files with matching filenames.
- Each JSON has a top-level field: {"text": "..."}.
- If "text" is missing / not a string / empty after strip, that pair is skipped.

Usage:
    python compute_meteor.py --gen_dir /path/to/gen_json --ref_dir /path/to/ref_json
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
from typing import Optional, List, Tuple

from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score


def load_json_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    text = data.get("text", None)
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text if text else None


def safe_word_tokenize(text: str) -> List[str]:
    """
    Robust tokenizer:
    1) Try NLTK word_tokenize (needs punkt/punkt_tab).
    2) If unavailable, fall back to a simple regex tokenizer.
    """
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except LookupError:
        # Fallback: split into words/punct using regex
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    except Exception:
        # Last resort
        return text.split()


def compute_batch_meteor_json(generated_dir: str, reference_dir: str) -> Tuple[float, List[Tuple[str, float]], dict]:
    total_score = 0.0
    count = 0
    individual_scores: List[Tuple[str, float]] = []

    stats = {
        "n_gen_json": 0,
        "n_missing_ref": 0,
        "n_gen_invalid_or_empty": 0,
        "n_ref_invalid_or_empty": 0,
        "n_valid_pairs": 0,
    }

    if not os.path.isdir(generated_dir):
        raise FileNotFoundError(f"generated_dir is not a directory: {generated_dir}")
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"reference_dir is not a directory: {reference_dir}")

    gen_files = sorted([fn for fn in os.listdir(generated_dir) if fn.endswith(".json")])
    stats["n_gen_json"] = len(gen_files)

    for filename in tqdm(gen_files, desc="Computing METEOR"):
        gen_path = os.path.join(generated_dir, filename)
        ref_path = os.path.join(reference_dir, filename)

        if not os.path.exists(ref_path):
            stats["n_missing_ref"] += 1
            continue

        hyp_text = load_json_text(gen_path)
        if hyp_text is None:
            stats["n_gen_invalid_or_empty"] += 1
            continue

        ref_text = load_json_text(ref_path)
        if ref_text is None:
            stats["n_ref_invalid_or_empty"] += 1
            continue

        hyp_tokens = safe_word_tokenize(hyp_text)
        ref_tokens = safe_word_tokenize(ref_text)

        sc = meteor_score([ref_tokens], hyp_tokens)
        individual_scores.append((filename, float(sc)))
        total_score += float(sc)
        count += 1

    avg_score = total_score / count if count > 0 else 0.0
    stats["n_valid_pairs"] = count
    return avg_score, individual_scores, stats


def main():
    parser = argparse.ArgumentParser(description="Compute METEOR from JSON files using top-level 'text'.")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated JSON files (*.json).")
    parser.add_argument("--ref_dir", required=True, help="Directory of reference JSON files (*.json).")
    args = parser.parse_args()

    # Download required NLTK resources (safe to call multiple times).
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)   # <-- IMPORTANT FIX for your error
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    avg, scores, stats = compute_batch_meteor_json(args.gen_dir, args.ref_dir)

    print("\n[Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== Individual METEOR scores ===")
    for fname, s in scores:
        print(f"{fname}: {s:.4f}")

    print(f"\n>>> Average METEOR score: {avg:.4f}")


if __name__ == "__main__":
    main()