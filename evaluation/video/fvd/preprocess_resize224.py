#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-resize MP4 videos to NxN using OpenCV.

Notes:
- Audio is NOT preserved (OpenCV VideoWriter typically writes video-only).
- Codec availability depends on your OpenCV build:
  try H.264 ('avc1'/'H264') first, fallback to 'mp4v'.

Example:
  python preprocess_resize224.py \
    --in_dir  ResponseNet/test/video/listener \
    --out_dir ResponseNet/test/video/listener224_cv \
    --size 224 --workers 8 --recursive
"""

import os
import cv2
import json
import time
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def list_mp4_files(in_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted([Path(p) for p in glob.glob(str(in_dir / pattern), recursive=recursive)])


def try_create_writer(out_path: Path, fps: float, size: int):
    """
    Try a few codecs. Return (writer, codec_str) or (None, None).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = ["mp4v"]  # order matters
    for c in candidates:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (size, size))
        if writer.isOpened():
            return writer, c
        try:
            writer.release()
        except Exception:
            pass
    return None, None


def transcode_one(job):
    in_path, out_path, size, target_fps, skip_existing = job
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return {"input": str(in_path), "output": str(out_path), "status": "skipped_exists", "seconds": 0.0}

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        return {"input": str(in_path), "output": str(out_path), "status": "failed_open", "seconds": time.time() - t0}

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0 or src_fps != src_fps:  # NaN check
        src_fps = 25.0

    out_fps = float(target_fps) if target_fps is not None else float(src_fps)

    writer, codec = try_create_writer(out_path, out_fps, size)
    if writer is None:
        cap.release()
        return {
            "input": str(in_path),
            "output": str(out_path),
            "status": "failed_writer_open",
            "seconds": time.time() - t0,
        }

    # Simple fps conversion:
    # - If target_fps is None: write every frame
    # - Else: use timestamp-based sampling
    frame_count_in = 0
    frame_count_out = 0

    if target_fps is None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count_in += 1
            frame_rs = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            writer.write(frame_rs)
            frame_count_out += 1
    else:
        # Timestamp-based resampling
        next_t = 0.0
        step = 1.0 / out_fps
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count_in += 1

            # position in seconds (may be 0 on some builds; fallback to frame index / src_fps)
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t_sec <= 0:
                t_sec = frame_count_in / float(src_fps)

            if t_sec + 1e-9 < next_t:
                continue

            frame_rs = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            writer.write(frame_rs)
            frame_count_out += 1
            next_t += step

    cap.release()
    writer.release()

    dt = time.time() - t0
    if not out_path.exists() or out_path.stat().st_size == 0 or frame_count_out == 0:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return {
            "input": str(in_path),
            "output": str(out_path),
            "status": "failed_empty_output",
            "seconds": dt,
            "src_fps": src_fps,
            "out_fps": out_fps,
            "frames_in": frame_count_in,
            "frames_out": frame_count_out,
            "codec": codec,
        }

    return {
        "input": str(in_path),
        "output": str(out_path),
        "status": "ok",
        "seconds": dt,
        "src_fps": src_fps,
        "out_fps": out_fps,
        "frames_in": frame_count_in,
        "frames_out": frame_count_out,
        "codec": codec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=None, help="Force output FPS (drops/keeps frames).")
    ap.add_argument("--no_skip_existing", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4s = list_mp4_files(in_dir, recursive=args.recursive)
    if not mp4s:
        print(f"[ERROR] No mp4 files found in: {in_dir}")
        return

    jobs = []
    for p in mp4s:
        rel = p.relative_to(in_dir) if args.recursive else Path(p.name)
        out_path = (out_dir / rel).with_suffix(".mp4")
        jobs.append((p, out_path, args.size, args.fps, (not args.no_skip_existing)))

    log_path = out_dir / "resize_log_opencv.jsonl"
    ok = failed = skipped = 0
    t_all = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(log_path, "a", encoding="utf-8") as f_log:
        futures = [ex.submit(transcode_one, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            f_log.write(json.dumps(r, ensure_ascii=False) + "\n")

            st = r.get("status", "unknown")
            if st == "ok":
                ok += 1
            elif st == "skipped_exists":
                skipped += 1
            else:
                failed += 1

            done = ok + skipped + failed
            if done % 20 == 0 or done == len(jobs):
                print(f"[PROGRESS] done={done}/{len(jobs)} ok={ok} skipped={skipped} failed={failed}")

    dt = time.time() - t_all
    print(f"[DONE] ok={ok}, skipped={skipped}, failed={failed}, total={len(jobs)}, seconds={dt:.1f}")
    print(f"[INFO] Log written to: {log_path}")


if __name__ == "__main__":
    # Prevent OpenCV from oversubscribing CPU when you already use multiprocessing
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    main()