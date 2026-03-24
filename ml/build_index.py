#!/usr/bin/env python3
"""
Build the FAISS image index from all downloaded puzzle screenshots.

Usage:
    python -m ml.build_index              # Build/update incrementally
    python -m ml.build_index --force      # Rebuild from scratch
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import List, Dict, Tuple
import argparse
import json
import sys
import numpy as np
from tqdm import tqdm

# Add parent dir to path so we can run as `python -m ml.build_index`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import (
    IMAGES_DIR, ANSWERS_FILE, ML_DIR,
    EMBEDDINGS_PATH, METADATA_PATH, INDEX_PATH,
    BATCH_SIZE, EMBED_DIM,
)
from ml.embeddings import load_model, embed_images


def load_answers():
    # type: () -> Dict[int, dict]
    with open(ANSWERS_FILE) as f:
        return {a["puzzle_num"]: a for a in json.load(f)}


def find_all_images(answers):
    # type: (Dict[int, dict]) -> Tuple[List[str], List[dict]]
    """Find all image paths and their metadata."""
    paths = []
    meta = []
    for puzzle_num in sorted(answers.keys()):
        puzzle_dir = os.path.join(IMAGES_DIR, str(puzzle_num))
        if not os.path.isdir(puzzle_dir):
            continue
        for img_num in range(1, 6):
            img_path = os.path.join(puzzle_dir, "%d.webp" % img_num)
            if os.path.exists(img_path):
                paths.append(img_path)
                meta.append({
                    "puzzle_num": puzzle_num,
                    "image_num": img_num,
                    "game_name": answers[puzzle_num]["answer"],
                    "developer": answers[puzzle_num].get("developer", ""),
                    "release_year": answers[puzzle_num].get("release_year", ""),
                })
    return paths, meta


def build_faiss_index(embeddings):
    # type: (np.ndarray) -> object
    """Build a FAISS inner-product index from normalized embeddings."""
    import faiss
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def save_index(embeddings, metadata, index):
    """Save all artifacts to disk."""
    import faiss; faiss.omp_set_num_threads(1)
    os.makedirs(ML_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    faiss.write_index(index, INDEX_PATH)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS image index")
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch")
    args = parser.parse_args()

    print("Loading answer database...")
    answers = load_answers()
    print("  %d puzzles in database" % len(answers))

    print("Scanning for images...")
    all_paths, all_meta = find_all_images(answers)
    print("  Found %d images across %d puzzles" % (
        len(all_paths),
        len(set(m["puzzle_num"] for m in all_meta)),
    ))

    # Check for incremental update
    existing_meta = []
    existing_embeddings = None
    if not args.force and os.path.exists(METADATA_PATH) and os.path.exists(EMBEDDINGS_PATH):
        with open(METADATA_PATH) as f:
            existing_meta = json.load(f)
        existing_embeddings = np.load(EMBEDDINGS_PATH)
        existing_keys = set(
            (m["puzzle_num"], m["image_num"]) for m in existing_meta
        )
        new_paths = []
        new_meta = []
        for path, meta in zip(all_paths, all_meta):
            key = (meta["puzzle_num"], meta["image_num"])
            if key not in existing_keys:
                new_paths.append(path)
                new_meta.append(meta)
        if not new_paths:
            print("Index is up to date! (%d embeddings)" % len(existing_meta))
            return
        print("  %d new images to embed (incremental update)" % len(new_paths))
        paths_to_embed = new_paths
        meta_to_embed = new_meta
    else:
        paths_to_embed = all_paths
        meta_to_embed = all_meta

    print("\nLoading CLIP model (ViT-B/32)...")
    print("  (First run will download ~350MB model weights)")
    model, preprocess = load_model()
    print("  Model loaded!")

    print("\nGenerating embeddings for %d images..." % len(paths_to_embed))
    print("  Batch size: %d" % BATCH_SIZE)

    # Process in chunks with progress bar
    all_new_embeddings = []
    valid_meta = []
    chunk_size = BATCH_SIZE * 4  # Process in larger chunks for progress reporting

    for start in tqdm(range(0, len(paths_to_embed), chunk_size), desc="Embedding"):
        end = min(start + chunk_size, len(paths_to_embed))
        chunk_paths = paths_to_embed[start:end]
        chunk_meta = meta_to_embed[start:end]

        embeddings, valid_indices = embed_images(chunk_paths, model, preprocess)
        all_new_embeddings.append(embeddings)
        for idx in valid_indices:
            valid_meta.append(chunk_meta[idx])

    new_embeddings = np.vstack(all_new_embeddings) if all_new_embeddings else np.zeros((0, EMBED_DIM), dtype=np.float32)

    # Merge with existing if incremental
    if existing_embeddings is not None and len(existing_meta) > 0:
        final_embeddings = np.vstack([existing_embeddings, new_embeddings])
        final_meta = existing_meta + valid_meta
    else:
        final_embeddings = new_embeddings
        final_meta = valid_meta

    print("\nBuilding FAISS index...")
    print("  Total embeddings: %d" % len(final_meta))
    print("  Unique games: %d" % len(set(m["game_name"] for m in final_meta)))
    index = build_faiss_index(final_embeddings)

    print("Saving to disk...")
    save_index(final_embeddings, final_meta, index)
    size_mb = (os.path.getsize(EMBEDDINGS_PATH) + os.path.getsize(INDEX_PATH)) / 1024 / 1024
    print("  Saved to %s/ (%.1f MB)" % (ML_DIR, size_mb))
    print("\nDone! Index ready for queries.")


if __name__ == "__main__":
    main()