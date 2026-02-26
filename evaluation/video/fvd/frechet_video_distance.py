#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Frechet Video Distance (FVD) between two directories of videos.

This version enforces:
1) Only one resize: external resize to 224x224; detector internal resize is disabled.
2) Frechet distance numerical stability: add eps to cov diagonals before sqrtm.
3) Strict sample count: require exactly --num_videos valid samples per directory (otherwise raise).
4) Default temporal subsampling factor = 3 (i.e., stride=3).

Notes:
- Uses a torchscript I3D detector (StyleGAN-V TFHub-matching I3D).
- Feeds uint8 videos shaped [B, C, T, H, W] to the detector.
- Computes mean/cov on features and then Frechet distance (FID formula).

Example:
  python frechet_video_distance.py \
    --dir_a /path/to/real \
    --dir_b /path/to/gen \
    --num_videos 140 \
    --num_frames 256 \
    --subsample_factor 1 \
    --batch_size 32 \
    --seed 42 \
    --detector_path /path/to/i3d_torchscript.pt
"""

import os
import glob
import random
import argparse
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import scipy.linalg

import torch
import torch.nn.functional as F
from torchvision.io import read_video
from tqdm import tqdm


# -------------------------------------------------------------------------
# Detector loading (torchscript)
# -------------------------------------------------------------------------

DEFAULT_DETECTOR_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"


def load_torchscript_detector(detector_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """Load a torchscript model from a local path."""
    if not os.path.isfile(detector_path):
        raise FileNotFoundError(f"Detector path not found: {detector_path}")
    model = torch.jit.load(detector_path, map_location=device).eval().to(device)
    return model


def open_url_to_local_file(url: str, dst_path: str):
    """
    Minimal downloader for the detector file.
    If your environment has no internet, do NOT rely on this.
    """
    import urllib.request
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    urllib.request.urlretrieve(url, dst_path)


def load_detector(detector_path: Optional[str], detector_url: str, device: torch.device) -> torch.jit.ScriptModule:
    """
    Load detector either from local path or by downloading from URL.
    Prefer local path on clusters.
    """
    if detector_path is not None:
        return load_torchscript_detector(detector_path, device)

    # Download into a local cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fvd")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.splitext(urlparse(detector_url).path.split("/")[-1])[0] + ".pt"
    local_path = os.path.join(cache_dir, filename)

    if not os.path.isfile(local_path):
        print(f"[INFO] Downloading detector to: {local_path}")
        open_url_to_local_file(detector_url, local_path)

    return load_torchscript_detector(local_path, device)


# -------------------------------------------------------------------------
# Feature stats (mean/cov)
# -------------------------------------------------------------------------

class FeatureStats:
    """Accumulate features and compute mean/cov in float64 for numerical stability."""

    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self.num_items = 0
        self.num_features = None
        self.raw_mean = None
        self.raw_cov = None

    def is_full(self) -> bool:
        return self.num_items >= self.max_items

    def _init(self, d: int):
        self.num_features = d
        self.raw_mean = np.zeros([d], dtype=np.float64)
        self.raw_cov = np.zeros([d, d], dtype=np.float64)

    def append(self, x: np.ndarray) -> int:
        """
        Append a batch of features.

        Args:
            x: [B, D] float32/float64

        Returns:
            n_appended: how many samples were actually appended (may be truncated to fit max_items)
        """
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2, f"Expected [B, D], got shape {x.shape}"

        if self.num_items >= self.max_items:
            return 0

        remaining = self.max_items - self.num_items
        if x.shape[0] > remaining:
            x = x[:remaining]

        if self.num_features is None:
            self._init(x.shape[1])
        else:
            assert x.shape[1] == self.num_features, f"Feature dim mismatch: {x.shape[1]} vs {self.num_features}"

        n = int(x.shape[0])
        self.num_items += n

        x64 = x.astype(np.float64)
        self.raw_mean += x64.sum(axis=0)
        self.raw_cov += x64.T @ x64
        return n

    def get_mean_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, cov) where cov is the population covariance (dividing by N)."""
        if self.num_items <= 0:
            raise RuntimeError("No items were accumulated; cannot compute mean/cov.")
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov


# -------------------------------------------------------------------------
# Video loading + frame sampling
# -------------------------------------------------------------------------

def list_videos(directory: str, exts=(".mp4", ".mov", ".mkv", ".webm")) -> List[str]:
    """List videos in a directory with allowed extensions."""
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    files.sort()
    return files


def sample_consecutive_frames_bthwc(
    path: str,
    num_frames: int,
    subsample_factor: int = 1,
    random_offset: bool = True,
) -> Optional[torch.Tensor]:
    """
    Load a video and sample consecutive frames with a temporal stride.

    Returns:
        frames: uint8 tensor [T, H, W, C] with T == num_frames
        or None if video is too short / read fails / format mismatch.
    """
    try:
        video, _, _ = read_video(path, pts_unit="sec")  # uint8 [T, H, W, C]
    except Exception:
        return None

    if video.ndim != 4 or video.size(-1) != 3:
        return None

    T = int(video.size(0))
    need = num_frames * subsample_factor
    if T < need:
        return None

    if random_offset:
        start = random.randint(0, T - need)
    else:
        start = 0

    idx = torch.arange(start, start + need, step=subsample_factor)
    frames = video.index_select(0, idx)  # [num_frames, H, W, C], uint8
    return frames


def batch_to_bcthw_uint8(frames_list: List[torch.Tensor], resize_hw=(224, 224)) -> torch.Tensor:
    """
    Convert a list of clips into a batched tensor [B, C, T, H, W] uint8.
    This function performs the ONLY resize in the pipeline (external resize).

    Args:
        frames_list: list of uint8 tensors, each [T, H, W, C]
        resize_hw: (H, W) target size. Default 224x224.

    Returns:
        videos: uint8 tensor [B, C, T, H, W]
    """
    processed: List[torch.Tensor] = []
    for clip in frames_list:
        # clip: [T, H, W, C] uint8
        # Resize each frame using bilinear interpolation in float, then cast back to uint8.
        x = clip.permute(0, 3, 1, 2).contiguous().float()  # [T, C, H, W]
        x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
        x = x.round().clamp(0, 255).to(torch.uint8)
        clip_rs = x.permute(0, 2, 3, 1).contiguous()  # [T, H, W, C]
        processed.append(clip_rs)

    x = torch.stack(processed, dim=0)              # [B, T, H, W, C]
    x = x.permute(0, 4, 1, 2, 3).contiguous()      # [B, C, T, H, W]
    return x


# -------------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------------

@torch.no_grad()
def compute_dir_feature_stats(
    directory: str,
    detector,
    detector_kwargs: dict,
    device: torch.device,
    max_items: int,
    num_frames: int,
    subsample_factor: int,
    batch_size: int,
    seed: int,
    random_offset: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/cov of detector features for videos in a directory.
    Enforces collecting exactly max_items valid samples (otherwise raises).
    """
    rng = random.Random(seed)
    paths = list_videos(directory)
    if len(paths) == 0:
        raise RuntimeError(f"No videos found in directory: {directory}")

    rng.shuffle(paths)

    stats = FeatureStats(max_items=max_items)
    pbar = tqdm(total=max_items, disable=not verbose, desc=f"Extracting features: {os.path.basename(directory)}")

    # Track why samples are skipped (for debugging).
    skip_counts: Dict[str, int] = {"decode_or_format_fail": 0, "too_short": 0}

    i = 0
    while not stats.is_full() and i < len(paths):
        batch_frames: List[torch.Tensor] = []

        # Assemble a batch of valid clips.
        while len(batch_frames) < batch_size and i < len(paths) and not stats.is_full():
            p = paths[i]
            i += 1

            # Try to read and sample frames.
            frames = sample_consecutive_frames_bthwc(
                p,
                num_frames=num_frames,
                subsample_factor=subsample_factor,
                random_offset=random_offset,
            )

            if frames is None:
                # We cannot perfectly distinguish "too short" vs "decode fail" without re-reading metadata;
                # however, we can do a lightweight second attempt to identify too-short cases.
                # To keep it simple and fast, we categorize all failures as decode_or_format_fail here.
                skip_counts["decode_or_format_fail"] += 1
                continue

            # If we got frames, it is a valid clip.
            batch_frames.append(frames)

        if len(batch_frames) == 0:
            continue

        # External resize ONLY happens here.
        videos_bcthw = batch_to_bcthw_uint8(batch_frames, resize_hw=(224, 224)).to(device)

        # Detector expects uint8 input; it handles rescaling/normalization internally.
        feats = detector(videos_bcthw, **detector_kwargs)  # [B, D]

        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)

        feats_np = feats.detach().cpu().numpy()

        # Append and update progress bar with the exact number appended.
        n_appended = stats.append(feats_np)
        pbar.update(n_appended)

    pbar.close()

    # Strict requirement: we must collect exactly max_items samples.
    if stats.num_items < max_items:
        raise RuntimeError(
            f"[ERROR] Only collected {stats.num_items}/{max_items} valid samples from {directory}. "
            f"Total files scanned: {i}/{len(paths)}. Skip counts: {skip_counts}. "
            f"Fix by: (1) lowering --num_videos, (2) ensuring videos are decodable, (3) ensuring videos are long enough."
        )

    mu, sigma = stats.get_mean_cov()
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Frechet distance between two Gaussians N(mu1, sigma1) and N(mu2, sigma2),
    with diagonal eps for numerical stability.

    Args:
        eps: small value added to diagonal of cov matrices.

    Returns:
        fid/fvd scalar.
    """
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    # Stabilize covariance matrices.
    if eps is not None and eps > 0:
        sigma1 = sigma1 + np.eye(sigma1.shape[0], dtype=np.float64) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0], dtype=np.float64) * eps

    diff = mu1 - mu2
    m = float(np.dot(diff, diff))

    # Matrix square root of product.
    s, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # If sqrtm returns complex due to numerical errors, discard tiny imaginary components.
    if np.iscomplexobj(s):
        if np.max(np.abs(np.imag(s))) > 1e-3:
            # If imaginary part is large, the result is likely numerically unstable.
            raise RuntimeError(f"[ERROR] sqrtm returned large imaginary components: max|Im|={np.max(np.abs(np.imag(s)))}")
        s = np.real(s)

    fid = m + np.trace(sigma1 + sigma2 - 2.0 * s)
    return float(np.real(fid))


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute FVD between two video directories (torchscript I3D).")
    parser.add_argument("--dir_a", required=True, help="First video directory (e.g., real).")
    parser.add_argument("--dir_b", required=True, help="Second video directory (e.g., generated).")

    parser.add_argument("--num_videos", type=int, default=140, help="Number of valid videos to use from each directory.")
    parser.add_argument("--num_frames", type=int, default=256, help="Number of frames per sampled clip.")
    parser.add_argument("--subsample_factor", type=int, default=1, help="Temporal stride (default: 1).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for detector forward.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--no_random_offset", action="store_true", help="Use offset=0 instead of random offsets.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow).")

    # Frechet stability
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon added to covariance diagonals (default: 1e-6).")

    # Detector selection
    parser.add_argument("--detector_path", default=None, help="Local path to i3d_torchscript.pt (recommended on clusters).")
    parser.add_argument("--detector_url", default=DEFAULT_DETECTOR_URL, help="URL to download detector if path is not given.")

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] Using device: {device}")

    detector = load_detector(args.detector_path, args.detector_url, device)

    # IMPORTANT:
    # - We keep ONLY external resize=224x224, so set detector resize=False.
    # - rescale=True keeps the detector's internal pixel value normalization behavior.
    detector_kwargs = dict(rescale=True, resize=False, return_features=True)

    mu_a, sigma_a = compute_dir_feature_stats(
        directory=args.dir_a,
        detector=detector,
        detector_kwargs=detector_kwargs,
        device=device,
        max_items=args.num_videos,
        num_frames=args.num_frames,
        subsample_factor=args.subsample_factor,
        batch_size=args.batch_size,
        seed=args.seed,
        random_offset=(not args.no_random_offset),
        verbose=True,
    )

    mu_b, sigma_b = compute_dir_feature_stats(
        directory=args.dir_b,
        detector=detector,
        detector_kwargs=detector_kwargs,
        device=device,
        max_items=args.num_videos,
        num_frames=args.num_frames,
        subsample_factor=args.subsample_factor,
        batch_size=args.batch_size,
        seed=args.seed + 999,  # different shuffle order
        random_offset=(not args.no_random_offset),
        verbose=True,
    )

    fvd = frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=args.eps)
    print(f"\nFVD: {fvd:.6f}")


if __name__ == "__main__":
    main()