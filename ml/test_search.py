#!/usr/bin/env python3
"""Quick test script for ML search."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys

# BASE is project root (parent of ml/)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

print("Importing...", flush=True)
from ml.embeddings import load_model, embed_images
print("Loading CLIP model...", flush=True)
model, preprocess = load_model()
print("Model loaded!", flush=True)

import json
import numpy as np
import faiss
faiss.omp_set_num_threads(1)

print("Loading FAISS index...", flush=True)
index = faiss.read_index(os.path.join(BASE, "data/ml/game_index.faiss"))
with open(os.path.join(BASE, "data/ml/metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
print("Index: %d vectors, %d metadata entries" % (index.ntotal, len(metadata)), flush=True)

# Search with all 5 images
puzzle = sys.argv[1] if len(sys.argv) > 1 else "1420"
paths = [os.path.join(BASE, "data/images/%s/%d.webp" % (puzzle, i)) for i in range(1, 6)]
paths = [p for p in paths if os.path.exists(p)]
print("Searching with %d images from puzzle %s..." % (len(paths), puzzle), flush=True)

embeddings, valid = embed_images(paths, model, preprocess)
print("Embedded %d images" % len(embeddings), flush=True)

k = 50
scores, indices = index.search(embeddings, k)

# Aggregate by game
game_scores = {}
for img_idx in range(len(embeddings)):
    per_image = {}
    for j in range(k):
        idx = indices[img_idx][j]
        if idx < 0:
            continue
        score = float(scores[img_idx][j])
        game = metadata[idx]["game_name"]
        if game not in per_image or score > per_image[game]:
            per_image[game] = score
    for game, score in per_image.items():
        if game not in game_scores:
            game_scores[game] = []
        game_scores[game].append(score)

# Rank by average score
ranked = sorted(
    game_scores.items(),
    key=lambda x: sum(x[1]) / len(embeddings),
    reverse=True,
)

print("\n" + "=" * 60)
print("  TOP 10 PREDICTIONS")
print("=" * 60)
for i, (game, scores_list) in enumerate(ranked[:10], 1):
    avg = sum(scores_list) / len(embeddings)
    mx = max(scores_list)
    conf = "HIGH" if avg > 0.85 else "MEDIUM" if avg > 0.75 else "LOW"
    print("  %d. %s" % (i, game))
    print("     Avg: %.4f  Max: %.4f  [%s]" % (avg, mx, conf))
print()