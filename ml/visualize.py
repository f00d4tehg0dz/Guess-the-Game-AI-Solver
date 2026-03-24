#!/usr/bin/env python3
"""
Generate a visual HTML report showing how the CLIP+FAISS pipeline identifies a game.

Produces a self-contained HTML file with:
  - Pipeline architecture diagram
  - Query images with embedding visualizations
  - Top FAISS matches with similarity score bars
  - Nearest-neighbor comparison grid (query vs. index match)
  - Embedding space PCA scatter plot

Usage:
    python -m ml.visualize 1410              # Visualize puzzle 1410
    python -m ml.visualize 1410 -o blog.html # Custom output filename
    python -m ml.visualize 1410 --top 10     # Show more matches
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import base64
import io
import json
import sys
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from ml.config import METADATA_PATH, INDEX_PATH, IMAGES_DIR, EMBED_DIM
from ml.embeddings import load_model, embed_images
from ml.solver import download_puzzle_images


def img_to_base64(path, max_size=400):
    """Convert an image file to a base64 data URI, resized for HTML embedding."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/webp;base64,{b64}"
    except Exception:
        return ""


def embedding_to_barcode(embedding, width=512, height=32):
    """Render a 512-dim embedding vector as a colored barcode image."""
    vals = np.array(embedding)
    # Normalize to 0-255 range for visualization
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin > 0:
        norm = ((vals - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        norm = np.full_like(vals, 128, dtype=np.uint8)

    # Create a colored barcode: blue (low) -> white (mid) -> red (high)
    img = np.zeros((height, len(norm), 3), dtype=np.uint8)
    for i, v in enumerate(norm):
        if v < 128:
            # Blue to white
            t = v / 128.0
            img[:, i] = [int(55 + 200 * t), int(55 + 200 * t), 255]
        else:
            # White to red
            t = (v - 128) / 127.0
            img[:, i] = [255, int(255 - 200 * t), int(255 - 200 * t)]

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((width, height), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def cosine_heatmap(query_embeddings, labels_q, match_embeddings, labels_m, width=500, height=400):
    """Render a cosine-similarity heatmap between query and match embeddings as base64 PNG."""
    # Compute similarity matrix
    sim = query_embeddings @ match_embeddings.T  # (Q, M)
    nq, nm = sim.shape

    cell_w = width // (nm + 1)
    cell_h = height // (nq + 1)
    img = np.full((height, width, 3), 24, dtype=np.uint8)  # dark bg

    for qi in range(nq):
        for mi in range(nm):
            v = float(sim[qi, mi])
            # Map 0.5..1.0 to color intensity
            t = max(0, min(1, (v - 0.5) / 0.5))
            r = int(20 + 235 * t)
            g = int(20 + 100 * t * (1 - t) * 4)
            b = int(20 + 50 * (1 - t))
            x0 = (mi + 1) * cell_w
            y0 = (qi + 1) * cell_h
            img[y0 + 1:y0 + cell_h - 1, x0 + 1:x0 + cell_w - 1] = [r, g, b]

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}", sim.tolist()


def pca_scatter(query_embs, match_embs, match_labels, width=600, height=400):
    """Simple 2D PCA projection rendered as inline SVG."""
    all_embs = np.vstack([query_embs, match_embs])
    # Center and project to 2D via PCA
    mean = all_embs.mean(axis=0)
    centered = all_embs - mean
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Top 2 eigenvectors (largest eigenvalues are last)
    proj = centered @ eigvecs[:, -2:]  # (N, 2)

    # Normalize to SVG coords
    margin = 40
    px = proj[:, 0]
    py = proj[:, 1]
    pmin, pmax = proj.min(), proj.max()
    rng = pmax - pmin if pmax - pmin > 0 else 1
    sx = margin + (px - pmin) / rng * (width - 2 * margin)
    sy = margin + (py - pmin) / rng * (height - 2 * margin)

    nq = len(query_embs)
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<rect width="{width}" height="{height}" fill="#111219" rx="8"/>'

    # Grid lines
    for gx in range(margin, width - margin, 50):
        svg += f'<line x1="{gx}" y1="{margin}" x2="{gx}" y2="{height - margin}" stroke="#222" stroke-width="0.5"/>'
    for gy in range(margin, height - margin, 50):
        svg += f'<line x1="{margin}" y1="{gy}" x2="{width - margin}" y2="{gy}" stroke="#222" stroke-width="0.5"/>'

    # Match points (smaller, semi-transparent)
    seen_labels = {}
    for i in range(nq, len(sx)):
        label = match_labels[i - nq] if i - nq < len(match_labels) else "?"
        color = _label_color(label, seen_labels)
        svg += f'<circle cx="{sx[i]:.1f}" cy="{sy[i]:.1f}" r="5" fill="{color}" opacity="0.6"/>'

    # Query points (larger, bright)
    for i in range(nq):
        svg += f'<circle cx="{sx[i]:.1f}" cy="{sy[i]:.1f}" r="9" fill="#00ff88" stroke="#fff" stroke-width="2"/>'
        svg += f'<text x="{sx[i] + 12:.1f}" y="{sy[i] + 4:.1f}" fill="#00ff88" font-size="11" font-family="monospace">Query {i + 1}</text>'

    # Legend (top matches only)
    unique_labels = list(dict.fromkeys(match_labels[:8]))
    for j, label in enumerate(unique_labels[:6]):
        color = _label_color(label, seen_labels)
        ly = margin + j * 18
        svg += f'<circle cx="{width - margin - 8}" cy="{ly}" r="5" fill="{color}"/>'
        short = label[:25] + "..." if len(label) > 25 else label
        svg += f'<text x="{width - margin - 18}" y="{ly + 4}" fill="#aaa" font-size="10" font-family="monospace" text-anchor="end">{short}</text>'

    svg += '</svg>'
    return svg


_COLOR_PALETTE = [
    "#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#4dabf7",
    "#9775fa", "#f783ac", "#66d9ef", "#e599f7", "#63e6be",
]

def _label_color(label, seen):
    if label not in seen:
        seen[label] = _COLOR_PALETTE[len(seen) % len(_COLOR_PALETTE)]
    return seen[label]


def generate_report(puzzle_num, top_n=5, output_path=None):
    """Run the full pipeline and generate an HTML visualization report."""
    print(f"Puzzle #{puzzle_num}")

    # Step 1: Download images
    print("Downloading puzzle images...")
    query_paths = download_puzzle_images(puzzle_num)
    print(f"  Got {len(query_paths)} images")
    if not query_paths:
        print("No images found!")
        return

    # Step 2: Load model
    print("Loading CLIP model...")
    model, preprocess = load_model()

    # Step 3: Embed query images
    print("Generating query embeddings...")
    query_embs, valid_indices = embed_images(query_paths, model, preprocess)
    query_paths = [query_paths[i] for i in valid_indices]

    # Step 4: Load index
    print("Loading FAISS index...")
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    # Step 5: Search
    print("Searching index...")
    k = min(top_n * 10, index.ntotal)
    scores, indices = index.search(query_embs, k)

    # Aggregate by game
    game_data = {}
    for img_idx in range(len(query_embs)):
        per_image = {}
        for j in range(k):
            idx = int(indices[img_idx][j])
            if idx < 0:
                continue
            score = float(scores[img_idx][j])
            meta = metadata[idx]
            game = meta["game_name"]
            if game not in per_image or score > per_image[game]["score"]:
                per_image[game] = {"score": score, "idx": idx, "meta": meta}
        for game, info in per_image.items():
            if game not in game_data:
                game_data[game] = {"scores": [], "best_indices": [], "meta": info["meta"]}
            game_data[game]["scores"].append(info["score"])
            game_data[game]["best_indices"].append(info["idx"])

    ranked = sorted(game_data.items(), key=lambda x: sum(x[1]["scores"]) / len(query_embs), reverse=True)

    # Collect data for visualization
    print("Building visualizations...")

    # Query image base64
    query_images_b64 = [img_to_base64(p) for p in query_paths]

    # Embedding barcodes for query images
    query_barcodes = [embedding_to_barcode(emb) for emb in query_embs]

    # Top match data with images from the index
    top_matches = []
    match_embs_list = []
    match_labels = []
    all_embeddings = np.load(os.path.join(os.path.dirname(INDEX_PATH), "embeddings.npy"))

    for game, data in ranked[:top_n]:
        avg_score = sum(data["scores"]) / len(query_embs)
        max_score = max(data["scores"])
        conf = "HIGH" if avg_score > 0.85 else "MEDIUM" if avg_score > 0.75 else "LOW"

        # Find best matching image from index for this game
        best_idx = data["best_indices"][np.argmax(data["scores"])]
        best_meta = metadata[best_idx]
        match_img_path = os.path.join(
            IMAGES_DIR, str(best_meta["puzzle_num"]),
            f"{best_meta['image_num']}.webp"
        )
        match_img_b64 = img_to_base64(match_img_path) if os.path.exists(match_img_path) else ""
        match_barcode = embedding_to_barcode(all_embeddings[best_idx]) if best_idx < len(all_embeddings) else ""

        # Collect embeddings for PCA
        for idx in data["best_indices"][:3]:
            if idx < len(all_embeddings):
                match_embs_list.append(all_embeddings[idx])
                match_labels.append(game)

        top_matches.append({
            "game": game,
            "avg_score": avg_score,
            "max_score": max_score,
            "confidence": conf,
            "per_image_scores": data["scores"],
            "match_img": match_img_b64,
            "match_barcode": match_barcode,
            "match_puzzle": best_meta["puzzle_num"],
            "match_image_num": best_meta["image_num"],
            "developer": data["meta"].get("developer", ""),
            "release_year": data["meta"].get("release_year", ""),
        })

    # PCA scatter
    match_embs_arr = np.array(match_embs_list) if match_embs_list else np.zeros((0, EMBED_DIM))
    pca_svg = pca_scatter(query_embs, match_embs_arr, match_labels)

    # Heatmap (query vs top match embeddings)
    heatmap_labels_m = [m["game"][:20] for m in top_matches[:8]]
    heatmap_embs = []
    for m in top_matches[:8]:
        idx = ranked[[g for g, _ in ranked].index(m["game"])][1]["best_indices"][0]
        if idx < len(all_embeddings):
            heatmap_embs.append(all_embeddings[idx])
    if heatmap_embs:
        heatmap_embs_arr = np.array(heatmap_embs)
        heatmap_b64, heatmap_data = cosine_heatmap(
            query_embs,
            [f"Img {i+1}" for i in range(len(query_embs))],
            heatmap_embs_arr,
            heatmap_labels_m,
        )
    else:
        heatmap_b64, heatmap_data = "", []

    # Build HTML
    print("Generating HTML report...")
    html = _build_html(
        puzzle_num=puzzle_num,
        query_images=query_images_b64,
        query_barcodes=query_barcodes,
        top_matches=top_matches,
        pca_svg=pca_svg,
        heatmap_img=heatmap_b64,
        heatmap_data=heatmap_data,
        total_indexed=index.ntotal,
        total_games=len(set(m["game_name"] for m in metadata)),
    )

    if output_path is None:
        output_path = os.path.join(BASE, f"visual_report_{puzzle_num}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to: {output_path}")
    print(f"Open in a browser to view the visualization.")
    return output_path


def _build_html(puzzle_num, query_images, query_barcodes, top_matches,
                pca_svg, heatmap_img, heatmap_data, total_indexed, total_games):
    """Build self-contained HTML report."""

    # Build query cards
    query_cards_html = ""
    for i, (img, barcode) in enumerate(zip(query_images, query_barcodes)):
        query_cards_html += f'''
        <div class="query-card">
          <div class="query-label">Clue {i + 1}</div>
          <img src="{img}" class="query-img" alt="Query image {i + 1}">
          <div class="barcode-label">512-dim CLIP Embedding</div>
          <img src="{barcode}" class="barcode" alt="Embedding barcode">
        </div>'''

    # Build match rows
    match_rows_html = ""
    for rank, m in enumerate(top_matches, 1):
        conf_class = m["confidence"].lower()
        score_pct = m["avg_score"] * 100
        max_pct = m["max_score"] * 100

        per_img_bars = ""
        for j, s in enumerate(m["per_image_scores"]):
            w = s * 100
            per_img_bars += f'<div class="mini-bar-wrap"><div class="mini-bar" style="width:{w:.0f}%"></div><span>Img {j+1}: {s:.3f}</span></div>'

        match_rows_html += f'''
        <div class="match-row {"top-match" if rank == 1 else ""}">
          <div class="match-rank">#{rank}</div>
          <div class="match-visual">
            <img src="{m['match_img']}" class="match-img" alt="{m['game']}">
            <div class="match-source">Puzzle #{m['match_puzzle']}, Image {m['match_image_num']}</div>
          </div>
          <div class="match-info">
            <div class="match-game">{m['game']}</div>
            <div class="match-meta">{m['developer']} {('(' + m['release_year'] + ')') if m['release_year'] else ''}</div>
            <div class="score-bar-container">
              <div class="score-bar" style="width: {score_pct:.0f}%">
                <span class="score-text">{m['avg_score']:.4f}</span>
              </div>
            </div>
            <div class="confidence-badge {conf_class}">{m['confidence']}</div>
            <div class="barcode-label" style="margin-top:8px">Best Match Embedding</div>
            <img src="{m['match_barcode']}" class="barcode" alt="Match embedding">
            <div class="per-image-detail">
              <div class="per-image-title">Per-Image Similarity</div>
              {per_img_bars}
            </div>
          </div>
        </div>'''

    # Heatmap cell annotations
    heatmap_section = ""
    if heatmap_img:
        heatmap_section = f'''
        <div class="section">
          <h2><span class="step-num">5</span> Cosine Similarity Heatmap</h2>
          <p class="desc">Each cell shows the cosine similarity between a query image embedding and a top-match embedding. Brighter red = higher similarity.</p>
          <div class="heatmap-wrap">
            <img src="{heatmap_img}" class="heatmap-img" alt="Similarity heatmap">
            <div class="heatmap-legend">
              <div class="heatmap-labels-y">
                {"".join(f'<div>Img {i+1}</div>' for i in range(len(query_images)))}
              </div>
            </div>
          </div>
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLIP + FAISS Visual Report — Puzzle #{puzzle_num}</title>
<style>
  :root {{
    --bg: #0d0f17;
    --surface: #151823;
    --surface2: #1c2030;
    --border: #2a2f45;
    --text: #e0e0e8;
    --text2: #8890a8;
    --accent: #00ff88;
    --accent2: #4dabf7;
    --red: #ff6b6b;
    --orange: #ffa94d;
    --yellow: #ffd43b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
    line-height: 1.6;
    padding: 0;
  }}
  .header {{
    background: linear-gradient(135deg, #0d0f17 0%, #1a1f35 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 32px 40px;
    text-align: center;
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }}
  .header .subtitle {{
    color: var(--text2);
    font-size: 14px;
    margin-top: 8px;
  }}
  .stats-row {{
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 24px;
    flex-wrap: wrap;
  }}
  .stat {{
    text-align: center;
  }}
  .stat-value {{
    font-size: 24px;
    font-weight: 700;
    color: var(--accent2);
  }}
  .stat-label {{
    font-size: 11px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px;
  }}
  .section {{
    margin-bottom: 48px;
  }}
  .section h2 {{
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--bg);
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
  }}
  .desc {{
    color: var(--text2);
    font-size: 13px;
    margin-bottom: 20px;
    max-width: 700px;
  }}

  /* Pipeline diagram */
  .pipeline {{
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 20px 0;
    justify-content: center;
    flex-wrap: wrap;
  }}
  .pipe-step {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    min-width: 120px;
    position: relative;
  }}
  .pipe-step.active {{
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
  }}
  .pipe-icon {{ font-size: 24px; margin-bottom: 6px; }}
  .pipe-label {{ font-size: 12px; font-weight: 600; }}
  .pipe-detail {{ font-size: 10px; color: var(--text2); margin-top: 2px; }}
  .pipe-arrow {{
    color: var(--accent);
    font-size: 20px;
    padding: 0 8px;
    flex-shrink: 0;
  }}

  /* Query images */
  .query-grid {{
    display: flex;
    gap: 16px;
    overflow-x: auto;
    padding-bottom: 8px;
  }}
  .query-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    min-width: 180px;
    flex-shrink: 0;
  }}
  .query-label {{
    font-size: 11px;
    color: var(--accent);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }}
  .query-img {{
    width: 170px;
    height: 130px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
  }}
  .barcode {{
    width: 100%;
    height: 24px;
    border-radius: 4px;
    margin-top: 4px;
    image-rendering: pixelated;
  }}
  .barcode-label {{
    font-size: 9px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 8px;
  }}

  /* Match results */
  .match-row {{
    display: flex;
    gap: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    align-items: flex-start;
    transition: border-color 0.2s;
  }}
  .match-row:hover {{
    border-color: var(--accent2);
  }}
  .match-row.top-match {{
    border-color: var(--accent);
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.08);
  }}
  .match-rank {{
    font-size: 20px;
    font-weight: 700;
    color: var(--text2);
    min-width: 40px;
    text-align: center;
    padding-top: 4px;
  }}
  .top-match .match-rank {{ color: var(--accent); }}
  .match-visual {{
    flex-shrink: 0;
    text-align: center;
  }}
  .match-img {{
    width: 160px;
    height: 120px;
    object-fit: cover;
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  .match-source {{
    font-size: 10px;
    color: var(--text2);
    margin-top: 4px;
  }}
  .match-info {{
    flex: 1;
    min-width: 0;
  }}
  .match-game {{
    font-size: 18px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 2px;
  }}
  .top-match .match-game {{ color: var(--accent); }}
  .match-meta {{
    font-size: 12px;
    color: var(--text2);
    margin-bottom: 10px;
  }}
  .score-bar-container {{
    background: var(--surface2);
    border-radius: 6px;
    height: 28px;
    overflow: hidden;
    position: relative;
  }}
  .score-bar {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    transition: width 0.5s ease;
    min-width: 60px;
  }}
  .score-text {{
    font-size: 12px;
    font-weight: 700;
    color: var(--bg);
  }}
  .confidence-badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 4px;
    margin-top: 8px;
    text-transform: uppercase;
  }}
  .confidence-badge.high {{ background: rgba(0,255,136,0.15); color: var(--accent); }}
  .confidence-badge.medium {{ background: rgba(255,169,77,0.15); color: var(--orange); }}
  .confidence-badge.low {{ background: rgba(255,107,107,0.15); color: var(--red); }}

  .per-image-detail {{
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }}
  .per-image-title {{
    font-size: 10px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }}
  .mini-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 3px;
  }}
  .mini-bar-wrap span {{
    font-size: 10px;
    color: var(--text2);
    min-width: 90px;
  }}
  .mini-bar {{
    height: 6px;
    background: var(--accent2);
    border-radius: 3px;
    flex: 1;
    max-width: 200px;
  }}

  /* PCA scatter */
  .scatter-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: inline-block;
  }}

  /* Heatmap */
  .heatmap-wrap {{
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }}
  .heatmap-img {{
    border-radius: 8px;
    max-width: 100%;
  }}
  .heatmap-labels-y {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 11px;
    color: var(--text2);
  }}

  /* Comparison grid */
  .compare-grid {{
    display: grid;
    grid-template-columns: auto repeat(auto-fill, minmax(160px, 1fr));
    gap: 8px;
    overflow-x: auto;
  }}
  .compare-header {{
    font-size: 10px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: center;
    padding: 8px 4px;
  }}
  .compare-cell {{
    text-align: center;
  }}
  .compare-cell img {{
    width: 100%;
    max-width: 150px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }}
  .compare-cell .score-tag {{
    font-size: 10px;
    color: var(--accent2);
    margin-top: 4px;
  }}

  .footer {{
    text-align: center;
    padding: 32px;
    color: var(--text2);
    font-size: 11px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}

  @media (max-width: 768px) {{
    .match-row {{ flex-direction: column; align-items: center; text-align: center; }}
    .query-grid {{ flex-wrap: nowrap; }}
    .pipeline {{ flex-direction: column; align-items: center; }}
    .pipe-arrow {{ transform: rotate(90deg); }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>CLIP + FAISS — Visual Pipeline Report</h1>
  <div class="subtitle">Puzzle #{puzzle_num} — How the AI identifies a game from screenshots</div>
  <div class="stats-row">
    <div class="stat">
      <div class="stat-value">{len(query_images)}</div>
      <div class="stat-label">Query Images</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total_indexed:,}</div>
      <div class="stat-label">Indexed Vectors</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total_games:,}</div>
      <div class="stat-label">Known Games</div>
    </div>
    <div class="stat">
      <div class="stat-value">512</div>
      <div class="stat-label">Embedding Dim</div>
    </div>
  </div>
</div>

<div class="container">

  <!-- Step 1: Pipeline overview -->
  <div class="section">
    <h2><span class="step-num">1</span> Pipeline Architecture</h2>
    <p class="desc">Each puzzle image is converted into a 512-dimensional vector by CLIP, then compared against an index of {total_indexed:,} pre-computed game screenshot embeddings via cosine similarity.</p>
    <div class="pipeline">
      <div class="pipe-step active">
        <div class="pipe-icon">🖼️</div>
        <div class="pipe-label">Download</div>
        <div class="pipe-detail">5 clue images</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step active">
        <div class="pipe-icon">👁️</div>
        <div class="pipe-label">CLIP ViT-B/32</div>
        <div class="pipe-detail">224×224 → 512-dim</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step active">
        <div class="pipe-icon">📐</div>
        <div class="pipe-label">L2 Normalize</div>
        <div class="pipe-detail">Unit sphere</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step active">
        <div class="pipe-icon">🔍</div>
        <div class="pipe-label">FAISS Search</div>
        <div class="pipe-detail">Inner product</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step active">
        <div class="pipe-icon">📊</div>
        <div class="pipe-label">Aggregate</div>
        <div class="pipe-detail">Score per game</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step active">
        <div class="pipe-icon">🏆</div>
        <div class="pipe-label">Rank</div>
        <div class="pipe-detail">Top predictions</div>
      </div>
    </div>
  </div>

  <!-- Step 2: Query images + embeddings -->
  <div class="section">
    <h2><span class="step-num">2</span> Query Images &amp; Embeddings</h2>
    <p class="desc">Each image is resized to 224×224 pixels and passed through CLIP's Vision Transformer. The output is a 512-dimensional vector (shown as a colored barcode: blue = negative, white = zero, red = positive).</p>
    <div class="query-grid">
      {query_cards_html}
    </div>
  </div>

  <!-- Step 3: FAISS results -->
  <div class="section">
    <h2><span class="step-num">3</span> Top Matches from FAISS Index</h2>
    <p class="desc">The query embeddings are compared against all {total_indexed:,} indexed screenshots. Scores are averaged across all query images to produce a final ranking. The bar shows similarity (1.0 = identical).</p>
    {match_rows_html}
  </div>

  <!-- Step 4: Embedding space -->
  <div class="section">
    <h2><span class="step-num">4</span> Embedding Space (PCA Projection)</h2>
    <p class="desc">All 512 dimensions projected down to 2D via PCA. Green dots are query images; colored dots are the nearest matches from the index. Clusters show visually similar games.</p>
    <div class="scatter-wrap">
      {pca_svg}
    </div>
  </div>

  {heatmap_section}

</div>

<div class="footer">
  Generated by Guess-the-Game AI Solver — CLIP ViT-B/32 + FAISS IndexFlatIP<br>
  Model: laion/CLIP-ViT-B-32-laion2B-s34B-b79K | Embedding dimension: 512
</div>

</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="Generate visual pipeline report")
    parser.add_argument("puzzle", type=int, help="Puzzle number to visualize")
    parser.add_argument("-o", "--output", help="Output HTML file path")
    parser.add_argument("--top", "-n", type=int, default=5, help="Number of top matches")
    args = parser.parse_args()
    generate_report(args.puzzle, top_n=args.top, output_path=args.output)


if __name__ == "__main__":
    main()
