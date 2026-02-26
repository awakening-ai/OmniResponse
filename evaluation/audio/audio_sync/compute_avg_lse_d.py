#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read all_scores.txt (or a given file) and print the average of the first column.

Each line is expected to have at least one number; the first column is LSE-D
(offset distance). The second column is typically confidence and is ignored
for the average.

Usage:
  python compute_avg_lse_d.py
  python compute_avg_lse_d.py /path/to/all_scores.txt
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Compute average of first column (LSE-D) from a scores file."
    )
    parser.add_argument(
        "scores_file",
        nargs="?",
        default="all_scores.txt",
        help="Path to file with two columns per line (default: all_scores.txt)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the numeric average, no label.",
    )
    args = parser.parse_args()

    path = Path(args.scores_file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                try:
                    values.append(float(parts[0]))
                except ValueError:
                    continue

    if not values:
        if not args.quiet:
            print("avg. LSE-D: (no valid scores)", file=sys.stderr)
        sys.exit(1)

    avg = sum(values) / len(values)
    if args.quiet:
        print(f"{avg:.6f}")
    else:
        print(f"avg. LSE-D: {avg:.6f}  (n={len(values)})")


if __name__ == "__main__":
    main()
