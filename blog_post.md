# I Built an AI That Identifies Video Games From Screenshots Using CLIP and FAISS

Every day, GuessThe.Game challenges players with five progressively easier screenshots from a video game. You get six guesses. Most people rely on memory and pattern recognition. I built a computer vision system that does it automatically.

Here's how I used OpenAI's CLIP model and Facebook's FAISS vector search to create a system that identifies games from screenshots alone — no API cheating, no answer databases at runtime. Pure visual recognition.

## The Problem

GuessThe.Game reveals five screenshots per puzzle, numbered easiest to hardest. The images could be anything: a close-up of a texture, a UI element, a character model, or a wide landscape shot. Humans draw on thousands of hours of gaming experience to recognize these. Can a machine do the same?

The answer is yes — but not in the way you might expect. This isn't traditional image classification with labeled categories. With over 1,400 unique games in the puzzle history, training a classifier for each game would be impractical. Instead, I turned it into a similarity search problem.

## The Architecture

The system works in two phases: indexing and querying.

### Phase 1: Building the Index

First, I scraped every historical puzzle — over 1,400 puzzles with 5 images each, totaling roughly 7,000+ screenshots. Each image gets processed through the same pipeline:

1. Load the image and resize to 224x224 pixels
2. Pass it through CLIP's Vision Transformer (ViT-B/32)
3. L2-normalize the resulting 512-dimensional embedding vector
4. Store the vector in a FAISS index alongside metadata (game name, puzzle number)

The result is a searchable database of ~7,000 embedding vectors, each one a compressed "fingerprint" of what CLIP sees in that screenshot.

### Phase 2: Solving a Puzzle

When a new puzzle drops, the solver downloads all five clue images and runs them through the exact same CLIP pipeline. Each image becomes a 512-dimensional query vector. Then FAISS kicks in — it compares each query vector against every vector in the index using cosine similarity (implemented as inner product on normalized vectors).

The scoring works like this: for each query image, find the best similarity score per game. Then average those scores across all five images. The game with the highest average similarity wins.

## Why CLIP?

CLIP (Contrastive Language-Image Pre-training) is a neural network trained by OpenAI on 400 million image-text pairs scraped from the internet. What makes it special for this task is that it doesn't just recognize objects — it understands visual concepts, art styles, UI patterns, and aesthetic qualities.

When CLIP processes a screenshot of, say, Hollow Knight, the embedding captures not just "dark cave with a character" but the distinctive art style, color palette, and visual language that makes Hollow Knight look like Hollow Knight. Two screenshots from the same game will produce similar embeddings even if they show completely different scenes.

I'm using the open-source variant from LAION — specifically ViT-B/32 trained on the LAION-2B dataset — via the open_clip library. The model is about 350MB and runs comfortably on both CPU and GPU.

### The Embedding

Here's what happens inside CLIP's Vision Transformer:

```python
def embed_images(image_paths, model, preprocess):
    batch = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img)  # Resize, normalize, etc.
        batch.append(tensor)

    batch_tensor = torch.stack(batch)
    with torch.no_grad():
        features = model.encode_image(batch_tensor)

    # L2 normalize — this is critical
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()
```

The preprocessing step is handled by CLIP's own transform pipeline: resize to 224x224, center crop, normalize pixel values to the distribution the model was trained on. The output is a 512-dimensional float32 vector with unit L2 norm.

That normalization step is crucial. By projecting every embedding onto the unit hypersphere, cosine similarity becomes equivalent to a simple dot product. This is what lets FAISS use its fast inner-product index.

### Visualizing the Embeddings

To understand what CLIP "sees," I render each 512-dimensional embedding as a colored barcode. Each of the 512 positions maps to a color: blue for negative values, white for near-zero, red for positive. Two screenshots from the same game produce visually similar barcodes — the patterns of activation align.

This is more than a gimmick. When you see two barcodes that look nearly identical, you're seeing that CLIP has extracted nearly identical high-level features from both images. The model has learned that these images share a visual identity.

## Why FAISS?

Facebook AI Similarity Search (FAISS) is a library purpose-built for searching through large collections of vectors. Even with a brute-force flat index (IndexFlatIP), searching 7,000 vectors takes under a millisecond.

```python
import faiss

# Build: just add normalized embeddings
index = faiss.IndexFlatIP(512)  # 512-dim inner product
index.add(embeddings)           # (N, 512) float32 array

# Search: find top 50 nearest neighbors per query
scores, indices = index.search(query_embeddings, k=50)
```

For this dataset size, a flat index is overkill-fast. If the index grew to millions of vectors, I'd switch to an IVF (Inverted File) index with clustering, trading a small amount of recall for orders-of-magnitude speedup.

### The Aggregation Strategy

Raw FAISS results give you per-image, per-vector similarity scores. The aggregation step is where the magic happens:

1. For each query image, find the highest similarity score for each game
2. Sum those best-per-image scores across all five query images
3. Divide by the number of query images to get an average
4. Rank games by average score

This approach is robust because it doesn't require every query image to match. If only three out of five screenshots strongly resemble a game's indexed images, that game still ranks high. It also handles the case where one very distinctive screenshot (like a title screen or unique UI) drives the match.

## The Scoring System

Empirical testing revealed clean confidence tiers:

- **HIGH** (score > 0.85): Almost certainly correct. The query images are visually near-identical to indexed images from the same game.
- **MEDIUM** (score 0.75–0.85): Strong match. Usually correct, but could be confused with visually similar games (e.g., games in the same franchise or engine).
- **LOW** (score < 0.75): Educated guess. The system found some visual similarity but isn't confident.

A score of 1.0 would mean pixel-perfect identity. In practice, scores above 0.90 typically mean the exact same scene was indexed from a previous puzzle.

## The Embedding Space

One fascinating aspect of CLIP embeddings is how they organize visual concepts. When you project the 512 dimensions down to 2D using PCA (Principal Component Analysis), games naturally cluster together.

Screenshot embeddings from the same game form tight clusters. Games with similar visual styles — like pixel art indie games, or realistic AAA shooters — form neighboring clusters. The embedding space essentially creates a map of visual similarity, and our search is just finding the nearest cluster to the query point.

This is what makes the system work even for games it's never seen from the same angle. CLIP's training on hundreds of millions of diverse images gave it a general understanding of visual similarity that transfers to game screenshots.

## Cross-Platform Challenges

One unexpected engineering challenge: Windows encoding. The metadata JSON file contains game names with special characters (accented letters, Japanese characters, etc.). macOS defaults to UTF-8 everywhere, but Windows Python defaults to cp1252, which chokes on these characters.

The fix is simple but easy to miss:

```python
# Windows-safe file reading
with open(metadata_path, encoding="utf-8") as f:
    metadata = json.load(f)
```

I also had to handle OpenMP thread conflicts between PyTorch and FAISS, which manifests as a cryptic crash on some systems:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
```

## Building the Index: The Scraping Pipeline

Before any ML happens, you need data. I built a scraping pipeline that:

1. Queries the GuessThe.Game API for historical answers
2. Downloads all five screenshots for each puzzle (parallel, with retry logic)
3. Stores images organized by puzzle number: `data/images/{puzzle_num}/{1-5}.webp`
4. Generates embeddings in batches of 64 images with progress tracking
5. Supports incremental updates — only embeds new images when puzzles are added

The full index build takes about 20 minutes on an M5 Pro and under 10 minutes on a 4090. The resulting artifacts are compact: ~15MB for embeddings, ~15MB for the FAISS index, and ~1MB for metadata.

## Results

On historical puzzles where the answer exists in the index, the system achieves HIGH confidence matches (>0.85) on roughly 70% of puzzles. MEDIUM confidence catches another 20%. The remaining 10% are games with very generic visuals or heavy stylistic overlap with other titles.

The system is weakest on:

- Pixel art games with similar palettes (Celeste vs. other precision platformers)
- Games from the same franchise with shared assets (different Call of Duty titles)
- Screenshots that show mostly text or menus

It's strongest on:

- Games with distinctive art styles (Okami, Cuphead, Hollow Knight)
- Screenshots showing unique UI elements or HUD designs
- Games with recognizable character models or environments

## The Visualization Tool

To make this all understandable, I built an HTML report generator that visualizes every step of the pipeline. For any puzzle, it produces a self-contained page showing:

- The five query images and their embedding barcodes
- Each top match with the closest indexed image side-by-side
- Per-image similarity breakdowns
- A PCA scatter plot of the embedding space
- A cosine similarity heatmap

This makes it possible to understand why the system made a particular prediction and where it went wrong.

## What's Next

The current system is a pure visual matcher — it can only identify games it has seen before. Future improvements could include:

- **Text recognition**: OCR on screenshots to catch game titles, UI text, and studio logos
- **Temporal analysis**: Using the video clue (clue 6 is a short WebM clip) for motion-based features
- **Ensemble methods**: Combining CLIP with game-specific classifiers for common genres
- **Fine-tuning**: Training a CLIP adapter specifically on game screenshot pairs

## Try It Yourself

The entire project is open source. To solve today's puzzle:

```bash
pip install -r requirements-ml.txt
python -m ml.solver
```

To generate a visual report:

```bash
python -m ml.visualize 1410 -o report.html
```

The core insight is that modern vision models like CLIP have already learned to see the way gamers do — they recognize style, composition, and visual identity, not just objects. Pair that with efficient vector search, and you get a system that can identify a game from a single screenshot in under a second.

The machines are getting good at our games. Time to make harder puzzles.
