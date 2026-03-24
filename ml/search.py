#!/usr/bin/env python3
"""
Search the FAISS index to identify a game from screenshots.

Usage:
    python -m ml.search data/images/1420/       # Identify from a puzzle dir
    python -m ml.search screenshot.png           # Single image
    python -m ml.search --top 10 image1.webp image2.webp
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import List, Optional
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import (
    METADATA_PATH, INDEX_PATH, EMBED_DIM,
)
from ml.embeddings import load_model, embed_images


def load_index():
    # type: () -> tuple
    """Load FAISS index and metadata."""
    import faiss; faiss.omp_set_num_threads(1)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def identify_game(image_paths, top_n=5, model=None, preprocess=None):
    # type: (List[str], int, Optional[object], Optional[object]) -> List[dict]
    """
    Identify a game from screenshot image paths.

    Returns list of dicts: [{"game": str, "score": float, "matches": [...]}]
    """
    if model is None or preprocess is None:
        model, preprocess = load_model()

    index, metadata = load_index()

    # Embed query images
    embeddings, valid_indices = embed_images(image_paths, model, preprocess)
    if len(embeddings) == 0:
        return []

    # Search FAISS — get top K per query image
    k = min(top_n * 10, index.ntotal)
    scores, indices = index.search(embeddings, k)

    # Aggregate scores by game across all query images
    game_scores = {}  # game_name -> {"total_score": float, "max_per_image": [...], "matches": [...]}

    for img_idx in range(len(embeddings)):
        image_game_scores = {}  # game_name -> best score for this image
        for j in range(k):
            idx = indices[img_idx][j]
            if idx < 0:
                continue
            score = float(scores[img_idx][j])
            meta = metadata[idx]
            game = meta["game_name"]

            if game not in image_game_scores or score > image_game_scores[game]:
                image_game_scores[game] = score

        for game, score in image_game_scores.items():
            if game not in game_scores:
                game_scores[game] = {
                    "total_score": 0.0,
                    "image_scores": [],
                    "matches": [],
                }
            game_scores[game]["total_score"] += score
            game_scores[game]["image_scores"].append(score)

    # Find matching puzzle info for top games
    for game, data in game_scores.items():
        data["avg_score"] = data["total_score"] / len(embeddings)
        matching_puzzles = set()
        for m in metadata:
            if m["game_name"] == game:
                matching_puzzles.add(m["puzzle_num"])
        data["matching_puzzles"] = sorted(matching_puzzles)

    # Sort by average score
    ranked = sorted(game_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    results = []
    for game, data in ranked[:top_n]:
        results.append({
            "game": game,
            "score": data["avg_score"],
            "max_score": max(data["image_scores"]),
            "matching_puzzles": data["matching_puzzles"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Identify a game from screenshots")
    parser.add_argument("paths", nargs="+", help="Image files or directory")
    parser.add_argument("--top", "-n", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    # Collect image paths
    image_paths = []
    for p in args.paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith((".webp", ".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(p, f))
        elif os.path.isfile(p):
            image_paths.append(p)
        else:
            print("Warning: %s not found, skipping" % p)

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print("Loading CLIP model...")
    model, preprocess = load_model()

    print("Analyzing %d images...\n" % len(image_paths))
    results = identify_game(image_paths, top_n=args.top, model=model, preprocess=preprocess)

    if not results:
        print("No matches found!")
        return

    print("=" * 60)
    print("  TOP %d PREDICTIONS" % len(results))
    print("=" * 60)
    for i, r in enumerate(results, 1):
        confidence = "HIGH" if r["score"] > 0.85 else "MEDIUM" if r["score"] > 0.75 else "LOW"
        print("  %d. %s" % (i, r["game"]))
        print("     Score: %.4f (max: %.4f) [%s]" % (r["score"], r["max_score"], confidence))
        print("     Seen in puzzle(s): %s" % ", ".join(str(p) for p in r["matching_puzzles"][:5]))
        print()


if __name__ == "__main__":
    main()