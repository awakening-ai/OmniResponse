#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import subprocess
import sys

def merge_video_audio(video_dir: Path, audio_dir: Path, output_dir: Path):
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all .mp4 files under video_dir (top-level only, no recursion)
    for video_path in video_dir.glob("*.mp4"):
        stem = video_path.stem
        audio_path = audio_dir / f"{stem}.wav"

        # Skip if the matching audio file does not exist
        if not audio_path.is_file():
            print(f"⚠️ Audio file not found: {audio_path}", file=sys.stderr)
            continue

        out_path = output_dir / f"{stem}.mp4"

        # Use ffmpeg to mux video + audio:
        # - copy the video stream (no re-encode)
        # - encode audio to AAC for broad compatibility
        cmd = [
            "ffmpeg",
            "-y",                  # Overwrite output if it already exists
            "-i", str(video_path), # Input video
            "-i", str(audio_path), # Input audio
            "-c:v", "copy",        # Copy video stream (no re-encoding)
            "-c:a", "aac",         # Encode audio as AAC
            "-b:a", "192k",        # Audio bitrate (adjust if needed)
            str(out_path),         # Output file
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            print(f"✅ Merged successfully: {out_path}")
        except subprocess.CalledProcessError:
            print(f"❌ Merge failed: {video_path} + {audio_path}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Merge same-named MP4 and WAV files into MP4 with audio"
    )
    parser.add_argument(
        "--video_dir", "-v",
        required=True,
        help="Directory containing MP4 files (top-level only, no recursion)",
    )
    parser.add_argument(
        "--audio_dir", "-a",
        required=True,
        help="Directory containing WAV files",
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Output directory for merged MP4 files",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    # Validate input directories
    if not video_dir.is_dir():
        print(f"Error: {video_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    merge_video_audio(video_dir, audio_dir, output_dir)

if __name__ == "__main__":
    main()