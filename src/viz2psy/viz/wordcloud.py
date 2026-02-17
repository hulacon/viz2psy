"""Word cloud visualization from CLIP embeddings."""

import fnmatch
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from wordcloud import WordCloud

# Default vocabulary for CLIP word clouds
# A mix of objects, scenes, actions, attributes, and abstract concepts
DEFAULT_VOCABULARY = [
    # Objects
    "person", "people", "man", "woman", "child", "baby", "face", "hand", "eye",
    "dog", "cat", "bird", "horse", "cow", "elephant", "bear", "lion", "fish",
    "car", "truck", "bicycle", "motorcycle", "airplane", "boat", "train", "bus",
    "tree", "flower", "grass", "forest", "mountain", "river", "ocean", "beach",
    "house", "building", "street", "road", "bridge", "tower", "church", "castle",
    "chair", "table", "bed", "couch", "lamp", "door", "window", "mirror",
    "food", "fruit", "vegetable", "cake", "pizza", "coffee", "wine", "bread",
    "book", "phone", "computer", "camera", "clock", "umbrella", "glasses",
    # Scenes
    "indoor", "outdoor", "city", "countryside", "urban", "rural", "suburban",
    "day", "night", "sunset", "sunrise", "morning", "evening", "afternoon",
    "summer", "winter", "spring", "autumn", "sunny", "cloudy", "rainy", "snowy",
    "kitchen", "bedroom", "bathroom", "office", "restaurant", "store", "park",
    "garden", "playground", "stadium", "museum", "library", "hospital", "school",
    # Attributes
    "beautiful", "ugly", "colorful", "bright", "dark", "light", "shadow",
    "large", "small", "tall", "short", "wide", "narrow", "deep", "shallow",
    "old", "new", "modern", "ancient", "vintage", "futuristic", "traditional",
    "clean", "dirty", "messy", "organized", "crowded", "empty", "peaceful",
    "happy", "sad", "angry", "calm", "excited", "relaxed", "tense", "serene",
    # Actions/States
    "walking", "running", "sitting", "standing", "sleeping", "eating", "drinking",
    "playing", "working", "reading", "writing", "talking", "laughing", "crying",
    "flying", "swimming", "climbing", "falling", "jumping", "dancing", "singing",
    # Abstract
    "nature", "technology", "art", "culture", "history", "science", "music",
    "love", "fear", "joy", "peace", "war", "freedom", "power", "beauty",
    "simple", "complex", "abstract", "realistic", "surreal", "minimal", "detailed",
    # Photography/Art terms
    "portrait", "landscape", "closeup", "wideangle", "bokeh", "silhouette",
    "reflection", "symmetry", "pattern", "texture", "contrast", "composition",
    # Colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "gold", "silver", "warm", "cool", "pastel", "vivid",
]


def get_clip_columns(df: pd.DataFrame) -> list[str]:
    """Get CLIP embedding column names from a DataFrame."""
    return [c for c in df.columns if c.startswith("clip_")]


def load_clip_text_encoder():
    """Load CLIP text encoder for computing text embeddings."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, tokenizer


def compute_text_embeddings(
    vocabulary: list[str],
    model,
    tokenizer,
    device: str = "cpu",
) -> np.ndarray:
    """Compute CLIP text embeddings for a vocabulary."""
    model = model.to(device)

    with torch.no_grad():
        # Tokenize all words/phrases
        tokens = tokenizer(vocabulary).to(device)
        # Get text embeddings
        text_features = model.encode_text(tokens)
        # L2 normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def compute_word_similarities(
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    vocabulary: list[str],
) -> dict[str, float]:
    """Compute cosine similarities between image and text embeddings."""
    # image_embedding should be (512,), text_embeddings should be (n_words, 512)
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    similarities = text_embeddings @ image_embedding

    # Convert to dict, filtering out negative similarities
    word_scores = {}
    for word, sim in zip(vocabulary, similarities):
        if sim > 0:
            word_scores[word] = float(sim)

    return word_scores


def make_wordcloud(
    df: pd.DataFrame,
    image_idx: int = 0,
    vocabulary: list[str] | None = None,
    top_n: int = 100,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    colormap: str = "viridis",
) -> plt.Figure:
    """Generate a word cloud from CLIP embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with CLIP embedding columns (clip_000, clip_001, ...).
    image_idx : int
        Row index of the image to visualize.
    vocabulary : list of str, optional
        Words to include in the cloud. Uses default vocabulary if not provided.
    top_n : int
        Maximum number of words to include.
    width, height : int
        Word cloud dimensions in pixels.
    background_color : str
        Background color.
    colormap : str
        Matplotlib colormap for word colors.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if vocabulary is None:
        vocabulary = DEFAULT_VOCABULARY

    # Extract CLIP embedding for the specified image
    clip_cols = get_clip_columns(df)
    if not clip_cols:
        raise ValueError("No CLIP columns found in DataFrame. Run viz2psy with 'clip' model first.")

    image_embedding = df.loc[image_idx, clip_cols].values.astype(np.float32)

    # Load CLIP text encoder and compute text embeddings
    model, tokenizer = load_clip_text_encoder()

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    text_embeddings = compute_text_embeddings(vocabulary, model, tokenizer, device)

    # Compute similarities
    word_scores = compute_word_similarities(image_embedding, text_embeddings, vocabulary)

    # Keep top N words
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    word_scores = dict(sorted_words)

    if not word_scores:
        raise ValueError("No words with positive similarity found.")

    # Generate word cloud
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=top_n,
    ).generate_from_frequencies(word_scores)

    # Create figure
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"CLIP Word Cloud (image {image_idx})")

    return fig
