#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ROUGE-L for a directory of generated JSON files against reference JSON files.

Assumptions:
- gen_dir and ref_dir both contain .json files with the SAME filenames.
- Each JSON file has a top-level field: {"text": "..."}.
- If "text" is missing / not a string / empty after strip, that sample will be skipped.

Usage example:
    python compute_rouge-l.py \
        --gen_dir /path/to/generated_jsons \
        --ref_dir /path/to/reference_jsons
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union, Optional


# ---------- ROUGE-L implementation (kept consistent with your version) ----------
def _lcs_len(a_tokens: List[str], b_tokens: List[str]) -> int:
    """Length of Longest Common Subsequence (LCS) between two token lists."""
    if len(a_tokens) < len(b_tokens):
        a_tokens, b_tokens = b_tokens, a_tokens

    m, n = len(a_tokens), len(b_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


class Rouge:
    """ROUGE-L scorer (Lin & Hovy, 2004)."""

    def __init__(self, beta: float = 1.2):
        self.beta = beta

    def _sentence_score(self, hyp: str, refs: List[str]) -> float:
        """Compute ROUGE-L for one hypothesis against multiple references."""
        hyp_tokens = hyp.split()
        prec, rec = 0.0, 0.0

        for r in refs:
            ref_tokens = r.split()
            lcs = _lcs_len(hyp_tokens, ref_tokens)
            prec = max(prec, lcs / max(len(hyp_tokens), 1))
            rec = max(rec, lcs / max(len(ref_tokens), 1))

        if prec > 0 and rec > 0:
            score = (1 + self.beta ** 2) * prec * rec / (rec + self.beta ** 2 * prec)
        else:
            score = 0.0
        return score

    def pair_score(self, hyp: str, ref: Union[str, List[str]]) -> float:
        """Compute ROUGE-L for hyp against one or multiple refs."""
        refs = ref if isinstance(ref, list) else [ref]
        return self._sentence_score(hyp, refs)


# ---------- JSON helpers ----------
def load_json_text(path: str) -> Optional[str]:
    """
    Load a JSON file and return the top-level 'text' field if valid.

    Returns:
        - stripped text (non-empty) if valid
        - None if missing / invalid / empty
    """
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
    if not text:
        return None

    return text


# ---------- Batch computation ----------
def compute_batch_rougeL_json(gen_dir: str, ref_dir: str) -> Tuple[float, List[Tuple[str, float]], dict]:
    """
    Compute average ROUGE-L over matched JSON files in gen_dir and ref_dir.

    Matching rule:
        - Only files ending with .json in gen_dir are considered.
        - Each gen file is matched to ref_dir/<same_filename>.
        - Samples with invalid/empty text in either side are skipped.

    Returns:
        avg_score: float
        indiv: list of (filename, score)
        stats: dict of counters
    """
    rouge = Rouge()
    filenames_scores: List[Tuple[str, float]] = []

    stats = {
        "n_gen_json": 0,
        "n_missing_ref": 0,
        "n_gen_invalid_or_empty": 0,
        "n_ref_invalid_or_empty": 0,
        "n_valid_pairs": 0,
    }

    gen_files = [fn for fn in os.listdir(gen_dir) if fn.endswith(".json")]
    gen_files.sort()
    stats["n_gen_json"] = len(gen_files)

    for fname in tqdm(gen_files, desc="Computing ROUGE-L"):
        gen_path = os.path.join(gen_dir, fname)
        ref_path = os.path.join(ref_dir, fname)

        if not os.path.exists(ref_path):
            stats["n_missing_ref"] += 1
            continue

        gen_txt = load_json_text(gen_path)
        if gen_txt is None:
            stats["n_gen_invalid_or_empty"] += 1
            continue

        ref_txt = load_json_text(ref_path)
        if ref_txt is None:
            stats["n_ref_invalid_or_empty"] += 1
            continue

        score = rouge.pair_score(gen_txt, ref_txt)
        filenames_scores.append((fname, float(score)))

    stats["n_valid_pairs"] = len(filenames_scores)

    scores = np.array([s for _, s in filenames_scores], dtype=float)
    avg_score = float(scores.mean()) if len(scores) else 0.0
    return avg_score, filenames_scores, stats


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE-L from JSON files using top-level 'text' field.")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated JSON files (*.json).")
    parser.add_argument("--ref_dir", required=True, help="Directory of reference JSON files (*.json).")
    args = parser.parse_args()

    avg, indiv, stats = compute_batch_rougeL_json(args.gen_dir, args.ref_dir)

    print("\n[Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== Individual ROUGE-L ===")
    for fname, sc in indiv:
        print(f"{fname}: {sc:.4f}")

    print(f"\n>>> Average ROUGE-L: {avg:.4f}")


if __name__ == "__main__":
    main()