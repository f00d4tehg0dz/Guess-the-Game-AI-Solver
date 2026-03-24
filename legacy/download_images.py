#!/usr/bin/env python3
"""Download puzzle images for a given puzzle number."""

from typing import Optional, List
import os
import sys
import requests

IMAGES_BASE = "https://images.guessthe.game/gtg_images"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VERSION = "2.3.75"


def download_puzzle_images(puzzle_num: int, output_dir: Optional[str] = None) -> List[str]:
    """Download all 5 screenshot images + video for a puzzle. Returns list of saved paths."""
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "images", str(puzzle_num))
    os.makedirs(output_dir, exist_ok=True)

    saved = []
    for i in range(1, 6):
        url = f"{IMAGES_BASE}/{puzzle_num}/{i}.webp?v={VERSION}"
        path = os.path.join(output_dir, f"{i}.webp")
        if os.path.exists(path):
            saved.append(path)
            continue
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                saved.append(path)
            else:
                print(f"  Image {i} not found (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  Error downloading image {i}: {e}")

    # Try video too
    video_url = f"{IMAGES_BASE}/{puzzle_num}/video/6.webm?v={VERSION}"
    video_path = os.path.join(output_dir, "6.webm")
    if not os.path.exists(video_path):
        try:
            resp = requests.get(video_url, timeout=15)
            if resp.status_code == 200:
                with open(video_path, "wb") as f:
                    f.write(resp.content)
                saved.append(video_path)
        except Exception:
            pass
    elif os.path.exists(video_path):
        saved.append(video_path)

    return saved


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_images.py <puzzle_num> [end_num]")
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start

    for num in range(start, end + 1):
        print(f"Downloading puzzle {num}...")
        paths = download_puzzle_images(num)
        print(f"  Saved {len(paths)} files")