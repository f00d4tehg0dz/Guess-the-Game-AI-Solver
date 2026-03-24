"""Shared configuration for the ML pipeline."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(DATA_DIR, "ml")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANSWERS_FILE = os.path.join(DATA_DIR, "answers.json")

EMBEDDINGS_PATH = os.path.join(ML_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(ML_DIR, "metadata.json")
INDEX_PATH = os.path.join(ML_DIR, "game_index.faiss")

# CLIP model config — ViT-B/32 is fast on CPU, 512-dim embeddings
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512
IMAGE_SIZE = 224
BATCH_SIZE = 64