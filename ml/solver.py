#!/usr/bin/env python3
"""
AI Vision Solver — identify today's game using CLIP + FAISS only.

Downloads the puzzle images and runs them through the ML pipeline.
No API answer cheating — pure computer vision.

Usage:
    python3 -m ml.solver              # Solve today's puzzle
    python3 -m ml.solver 1420         # Solve a specific puzzle
    python3 -m ml.solver --top 10     # Show more predictions
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import tempfile
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

IMAGES_BASE = "https://images.guessthe.game/gtg_images"
VERSION = "2.3.75"


def find_today_puzzle():
    """Binary search for the latest puzzle number."""
    lo, hi = 1, 2000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            resp = requests.head(f"{IMAGES_BASE}/{mid}/1.webp?v={VERSION}", timeout=10)
            if resp.status_code == 200:
                lo = mid
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
    return lo


def download_puzzle_images(puzzle_num):
    """Download images to a temp dir, return list of paths."""
    tmpdir = tempfile.mkdtemp(prefix=f"gtg_{puzzle_num}_")
    paths = []
    for i in range(1, 6):
        url = f"{IMAGES_BASE}/{puzzle_num}/{i}.webp?v={VERSION}"
        path = os.path.join(tmpdir, f"{i}.webp")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                paths.append(path)
        except Exception:
            pass
    return paths


def solve(puzzle_num=None, top_n=5):
    from ml.search import identify_game
    from ml.embeddings import load_model

    if puzzle_num is None:
        print("Finding today's puzzle...")
        puzzle_num = find_today_puzzle()

    print(f"Puzzle #{puzzle_num}")
    print(f"Downloading images...")
    paths = download_puzzle_images(puzzle_num)
    print(f"  Got {len(paths)} images")

    if not paths:
        print("No images found!")
        return None

    print("Loading CLIP model...")
    model, preprocess = load_model()

    print("Analyzing images...\n")
    results = identify_game(paths, top_n=top_n, model=model, preprocess=preprocess)

    if not results:
        print("No matches found!")
        return None

    print("=" * 55)
    print("  PREDICTIONS")
    print("=" * 55)
    for i, r in enumerate(results, 1):
        conf = "HIGH" if r["score"] > 0.85 else "MED" if r["score"] > 0.75 else "LOW"
        print(f"  {i}. {r['game']}")
        print(f"     Score: {r['score']:.4f}  Max: {r['max_score']:.4f}  [{conf}]")
    print("=" * 55)

    best = results[0]
    print(f"\n  ANSWER: {best['game']}  (confidence: {best['score']:.4f})\n")
    return best["game"]


def main():
    parser = argparse.ArgumentParser(description="AI Vision Solver")
    parser.add_argument("puzzle", nargs="?", type=int, help="Puzzle number (default: today)")
    parser.add_argument("--top", "-n", type=int, default=5, help="Number of predictions")
    args = parser.parse_args()
    solve(args.puzzle, args.top)


if __name__ == "__main__":
    main()
