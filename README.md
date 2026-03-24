# GuessThe.Game Cheat

AI-powered solver for [guessthe.game](https://guessthe.game/) — identifies video games from screenshot clues using CLIP embeddings and FAISS similarity search.

## How It Works

1. **Download** all historical puzzle images (1–1400+)
2. **Scrape** the answer for each puzzle from the API
3. **Embed** every screenshot with OpenAI CLIP (ViT-B/32) into 512-dim vectors
4. **Index** embeddings in FAISS for fast nearest-neighbor search
5. **Solve** new puzzles by embedding their screenshots and matching against the index

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt
pip3 install -r requirements-ml.txt

# Build the answer database
python3 -m legacy.scrape_answers

# Download all puzzle images
python3 -m legacy.download_all_images

# Build the FAISS index (first run downloads ~350MB CLIP model)
python3 -m ml.build_index

# Solve today's puzzle with AI vision
python3 -m ml.solver
```

## Project Structure

```
ml/                     # Machine vision pipeline
  solver.py             # AI vision solver — download images, predict, print answer
  search.py             # FAISS index search — identify game from image paths
  build_index.py        # Build/update FAISS index from downloaded screenshots
  embeddings.py         # CLIP model loading and image embedding
  config.py             # Shared paths and model config
  test_search.py        # Quick test script for the ML pipeline

legacy/                 # API-based tools (no ML)
  cheat.py              # All-in-one cheat CLI (answer lookup, submit, scrape)
  solver.py             # Simple solver using the game_info API exploit
  scrape_answers.py     # Scrape all puzzle answers from the API
  download_images.py    # Download images for a specific puzzle
  download_all_images.py  # Bulk parallel image downloader

data/                   # Data files (not committed)
  answers.json          # Scraped answer database
  images/               # Downloaded puzzle screenshots
  ml/                   # FAISS index, embeddings, metadata
```

## Usage

### Solve with AI Vision (pure ML, no answer API)

```bash
python3 -m ml.solver              # Today's puzzle
python3 -m ml.solver 1420         # Specific puzzle
python3 -m ml.solver --top 10     # More predictions
```

### Solve with API Exploit (instant, no ML needed)

```bash
python3 -m legacy.cheat                    # Show today's answer
python3 -m legacy.cheat --submit           # Auto-submit the answer
python3 -m legacy.cheat --puzzle 1409      # Specific puzzle
```

### Build / Update the Index

```bash
# Scrape any new answers
python3 -m legacy.scrape_answers

# Download new images (parallel, 20 threads)
python3 -m legacy.download_all_images

# Update the FAISS index incrementally
python3 -m ml.build_index

# Or rebuild from scratch
python3 -m ml.build_index --force
```

### Search from Custom Images

```bash
python3 -m ml.search data/images/1420/         # From a puzzle directory
python3 -m ml.search screenshot.png             # Single image
python3 -m ml.search --top 10 img1.webp img2.webp
```

## Requirements

- Python 3.9+
- **Core:** `requests`, `Pillow`
- **ML:** `torch`, `open-clip-torch`, `faiss-cpu`, `numpy`, `tqdm`

## The Exploit

The `game_info` API at `https://api.guessthe.game/api/game_info/?puzzle_num=N&puzzle_type=gtg` returns the answer for any puzzle — past or present — with no authentication. The legacy tools use this directly. The ML pipeline exists for when you want to solve it the hard way.
