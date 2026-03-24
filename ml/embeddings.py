"""Core CLIP model loading and image embedding extraction."""

from typing import List, Tuple
import numpy as np
import torch
from PIL import Image

from .config import MODEL_NAME, PRETRAINED, BATCH_SIZE


def load_model():
    # type: () -> Tuple
    """Load CLIP visual encoder. Returns (model, preprocess_fn)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()
    return model, preprocess


def embed_images(image_paths, model, preprocess, batch_size=BATCH_SIZE):
    # type: (List[str], object, object, int) -> Tuple[np.ndarray, List[int]]
    """
    Generate normalized CLIP embeddings for a list of images.

    Returns:
        embeddings: (N, 512) float32 numpy array of L2-normalized vectors
        valid_indices: list of indices into image_paths that were successfully processed
    """
    all_embeddings = []
    valid_indices = []
    batch = []
    batch_indices = []

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img)
            batch.append(tensor)
            batch_indices.append(i)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

        if len(batch) >= batch_size:
            embs = _encode_batch(batch, model)
            all_embeddings.append(embs)
            valid_indices.extend(batch_indices)
            batch = []
            batch_indices = []

    # Remaining batch
    if batch:
        embs = _encode_batch(batch, model)
        all_embeddings.append(embs)
        valid_indices.extend(batch_indices)

    if not all_embeddings:
        return np.zeros((0, 512), dtype=np.float32), []

    embeddings = np.vstack(all_embeddings)
    return embeddings, valid_indices


def _encode_batch(tensors, model):
    # type: (list, object) -> np.ndarray
    """Encode a batch of preprocessed image tensors through CLIP."""
    batch_tensor = torch.stack(tensors)
    with torch.no_grad():
        features = model.encode_image(batch_tensor)
    # L2 normalize for cosine similarity via inner product
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)