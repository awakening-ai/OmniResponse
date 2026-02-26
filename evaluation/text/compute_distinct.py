#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional


def get_ngrams(tokens: List[str], n: int):
    """Return a list of n-grams (as tuples) from a token list."""
    return list(zip(*[tokens[i:] for i in range(n)]))


def compute_distinct(hypotheses: List[str]) -> Tuple[float, float]:
    """
    Compute Distinct-1 and Distinct-2 for a list of hypothesis strings.

    Distinct-1 = (# unique unigrams) / (# total unigrams)
    Distinct-2 = (# unique bigrams)  / (# total bigrams)
    """
    total_unigrams = 0
    total_bigrams = 0
    unique_unigrams = set()
    unique_bigrams = set()

    for hyp in hypotheses:
        tokens = hyp.strip().split()
        if not tokens:
            continue

        total_unigrams += len(tokens)
        total_bigrams += max(len(tokens) - 1, 0)

        unique_unigrams.update(tokens)
        unique_bigrams.update(get_ngrams(tokens, 2))

    distinct1 = len(unique_unigrams) / total_unigrams if total_unigrams > 0 else 0.0
    distinct2 = len(unique_bigrams) / total_bigrams if total_bigrams > 0 else 0.0
    return distinct1, distinct2


def _find_text_field_recursively(obj) -> Optional[str]:
    """
    Recursively search for the first string-valued field named 'text'.

    This is an optional fallback for JSON files whose structure is not fixed.
    """
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            return obj["text"]
        for v in obj.values():
            r = _find_text_field_recursively(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_text_field_recursively(item)
            if r is not None:
                return r
    return None


def load_hypotheses_json(generated_dir: str, *, use_recursive_fallback: bool = False):
    """
    Load hypotheses from all .json files under `generated_dir`.

    Rules:
    - By default, read the top-level field data["text"].
    - If "text" is missing / not a string / empty after strip: skip the file.
    - If `use_recursive_fallback=True`, and top-level "text" is missing/invalid,
      try to recursively search for a string field named "text".

    Returns:
      hyps:  List[str] - collected non-empty hypothesis texts
      stats: dict      - counters for debugging/monitoring data quality
    """
    stats = {
        "n_files_total": 0,        # total number of files in the directory
        "n_json_files": 0,         # number of .json files
        "n_parse_fail": 0,         # JSON parsing failures
        "n_missing_text": 0,       # missing top-level "text"
        "n_text_not_str": 0,       # top-level "text" exists but is not a string (or is None)
        "n_empty_text": 0,         # "text" is empty after stripping
        "n_used_fallback": 0,      # how many times recursive fallback was used
        "n_valid": 0,              # number of valid hypotheses collected
    }

    if not os.path.isdir(generated_dir):
        raise FileNotFoundError(f"--gen_dir is not a directory: {generated_dir}")

    all_files = os.listdir(generated_dir)
    stats["n_files_total"] = len(all_files)

    json_files = [fn for fn in all_files if fn.endswith(".json")]
    json_files.sort()
    stats["n_json_files"] = len(json_files)

    hyps: List[str] = []

    for filename in tqdm(json_files, desc="Loading JSON"):
        path = os.path.join(generated_dir, filename)

        # Parse JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            stats["n_parse_fail"] += 1
            continue

        text = None
        used_fallback = False

        # Prefer top-level "text"
        if isinstance(data, dict) and "text" in data:
            text = data.get("text", None)
        else:
            stats["n_missing_text"] += 1

        # If top-level text is invalid, optionally try recursive fallback
        if not isinstance(text, str):
            # Count cases where "text" exists but is not a string
            if isinstance(data, dict) and "text" in data:
                stats["n_text_not_str"] += 1

            if use_recursive_fallback:
                cand = _find_text_field_recursively(data)
                if isinstance(cand, str):
                    text = cand
                    used_fallback = True

        # Still invalid -> skip
        if not isinstance(text, str):
            continue

        # Strip and filter empty
        text = text.strip()
        if not text:
            stats["n_empty_text"] += 1
            continue

        if used_fallback:
            stats["n_used_fallback"] += 1

        hyps.append(text)

    stats["n_valid"] = len(hyps)
    return hyps, stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute Distinct-1/2 from a directory of JSON files (using field 'text')."
    )
    parser.add_argument("--gen_dir", type=str, required=True, help="Input directory containing .json files.")
    parser.add_argument(
        "--recursive_fallback",
        action="store_true",
        help="If top-level 'text' is missing/non-string, recursively search for a string 'text' field.",
    )
    args = parser.parse_args()

    hypotheses, stats = load_hypotheses_json(args.gen_dir, use_recursive_fallback=args.recursive_fallback)
    distinct1, distinct2 = compute_distinct(hypotheses)

    print("\n[Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\n>>> Distinct-1: {distinct1:.4f}")
    print(f">>> Distinct-2: {distinct2:.4f}")


if __name__ == "__main__":
    main()