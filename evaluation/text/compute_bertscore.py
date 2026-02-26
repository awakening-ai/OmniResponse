#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute BERTScore for a directory of generated JSON files against reference JSON files.

Assumptions:
- gen_dir and ref_dir both contain .json files with matching filenames.
- Each JSON has a top-level field: {"text": "..."}.
- If "text" is missing / not a string / empty after strip, that pair is skipped.

Usage:
    python compute_bertscore.py --gen_dir /path/to/gen_json --ref_dir /path/to/ref_json
"""

import os
import json
import argparse
from typing import Optional, List, Tuple

from bert_score import score
from tqdm import tqdm


def load_json_text(path: str) -> Optional[str]:
    """Load a JSON file and return the top-level 'text' if it is a non-empty string; otherwise None."""
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


def compute_batch_bertscore_json(
    generated_dir: str,
    reference_dir: str,
    lang: str = "en",
    model_type: Optional[str] = None,
    batch_size: int = 64,
    num_layers: Optional[int] = None,
    rescale_with_baseline: bool = False,
) -> Tuple[float, float, float, float, List[Tuple[str, float]], dict]:
    """
    Compute BERTScore P/R/F1 for matched JSON files in generated_dir vs reference_dir.

    Returns:
        avg_f1, avg_p, avg_r, n_valid, results(list of (filename, f1)), stats(dict)
    """
    hyps: List[str] = []
    refs: List[str] = []
    filenames: List[str] = []

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

    gen_files = [fn for fn in os.listdir(generated_dir) if fn.endswith(".json")]
    gen_files.sort()
    stats["n_gen_json"] = len(gen_files)

    for filename in tqdm(gen_files, desc="Loading pairs"):
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

        hyps.append(hyp_text)
        refs.append(ref_text)
        filenames.append(filename)

    stats["n_valid_pairs"] = len(filenames)

    if not hyps:
        # No valid pairs
        return 0.0, 0.0, 0.0, 0.0, [], stats

    # Compute BERTScore
    P, R, F1 = score(
        cands=hyps,
        refs=refs,
        lang=lang,
        model_type=model_type,
        batch_size=batch_size,
        num_layers=num_layers,
        rescale_with_baseline=rescale_with_baseline,
        verbose=True,
    )

    f1_list = F1.tolist()
    results = list(zip(filenames, f1_list))

    avg_p = P.mean().item()
    avg_r = R.mean().item()
    avg_f1 = F1.mean().item()
    n_valid = float(len(filenames))

    return avg_f1, avg_p, avg_r, n_valid, results, stats


def main():
    parser = argparse.ArgumentParser(description="Compute BERTScore from JSON files using top-level 'text'.")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated JSON files (*.json).")
    parser.add_argument("--ref_dir", required=True, help="Directory of reference JSON files (*.json).")
    parser.add_argument("--lang", default="en", help="Language code for BERTScore (default: en).")
    parser.add_argument("--model_type", default=None, help="Optional HuggingFace model name for BERTScore.")
    parser.add_argument("--batch_size", type=int, default=64, help="BERTScore batch size (default: 64).")
    parser.add_argument("--num_layers", type=int, default=None, help="Optional: use first N layers of the model.")
    parser.add_argument(
        "--rescale_with_baseline",
        action="store_true",
        help="Rescale BERTScore with baseline (may download baseline files).",
    )
    args = parser.parse_args()

    avg_f1, avg_p, avg_r, n_valid, individual_scores, stats = compute_batch_bertscore_json(
        generated_dir=args.gen_dir,
        reference_dir=args.ref_dir,
        lang=args.lang,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        rescale_with_baseline=args.rescale_with_baseline,
    )

    print("\n[Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== Individual BERTScore-F1 ===")
    for fname, s in individual_scores:
        print(f"{fname}: {s:.4f}")

    print(f"\n>>> Average BERTScore-P:  {avg_p:.4f}")
    print(f">>> Average BERTScore-R:  {avg_r:.4f}")
    print(f">>> Average BERTScore-F1: {avg_f1:.4f}")
    print(f">>> #Valid pairs:         {int(n_valid)}")


if __name__ == "__main__":
    main()