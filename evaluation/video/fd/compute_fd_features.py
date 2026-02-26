#!/usr/bin/env python3
"""
compute_fid_features.py

Compute the Fréchet Distance (FID) between two sets of frame‐level feature vectors
stored as .npy files (shape: T×C) in two directories.

Usage:
    python compute_fid_features.py --pred_dir /path/to/predictions \
                                   --gt_dir   /path/to/ground_truth
"""

import os
import argparse
import warnings

import numpy as np
from scipy import linalg

def load_and_stack(dir_path):
    """
    Recursively loads all .npy files under dir_path, each of shape (T, C),
    and concatenates them into a single array of shape (sum_i T_i, C).
    """
    feats = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not fname.endswith('.npy'):
                continue
            path = os.path.join(root, fname)
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError(f"Expected a 2D array in '{path}', got shape {arr.shape}")
            feats.append(arr)
    if not feats:
        raise RuntimeError(f"No .npy feature files found in '{dir_path}'")
    return np.concatenate(feats, axis=0)

def calculate_activation_statistics(feats):
    """
    Given an array feats of shape (N, C), compute:
      mu    = mean vector of shape (C,)
      sigma = covariance matrix of shape (C, C)
    """
    mu    = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Fréchet Distance between two Gaussians.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Covariance matrices have different dimensions")

    diff = mu1 - mu2
    # Product might be nearly singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(
            "FID calculation produces singular product; "
            f"adding {eps} to diagonal of covariances"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)

def main():
    parser = argparse.ArgumentParser(
        description="Compute FID between two directories of frame‐level feature .npy files"
    )
    parser.add_argument(
        "--pred_dir", required=True,
        help="Directory with predicted .npy feature files"
    )
    parser.add_argument(
        "--gt_dir", required=True,
        help="Directory with ground‐truth .npy feature files"
    )
    args = parser.parse_args()

    print(f"Loading features from: {args.pred_dir}")
    feats_pred = load_and_stack(args.pred_dir)
    print(f"Loading features from: {args.gt_dir}")
    feats_gt = load_and_stack(args.gt_dir)

    print("Calculating statistics...")
    mu1, sigma1 = calculate_activation_statistics(feats_pred)
    mu2, sigma2 = calculate_activation_statistics(feats_gt)

    print("Computing Fréchet Distance (FID)...")
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID = {fid_value:.6f}")

if __name__ == "__main__":
    main()
