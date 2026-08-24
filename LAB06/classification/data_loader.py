import os
from pathlib import Path
import numpy as np
import pandas as pd

# Supported relative paths to locate the Spotify dataset across different lab folder setups
POSSIBLE_PATHS = [
    Path(__file__).resolve().parent.parent / "data-music" / "spotify-2023.csv",
    Path(__file__).resolve().parent.parent.parent / "LAB04" / "data-music" / "spotify-2023.csv",
    Path(__file__).resolve().parent.parent.parent / "LAB05" / "data-music" / "spotify-2023.csv",
    Path("data-music/spotify-2023.csv"),
]

TARGET = "mode"

# Continuous audio feature list
NUMERIC_FEATURES = [
    "bpm",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]

# Categorical mapping for musical key notations
TEXT_FEATURES = {
    "key": {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }
}


def find_dataset_file(data_path=None):
    """Resolves and validates the dataset filepath."""
    if data_path and Path(data_path).exists():
        return Path(data_path)
    for p in POSSIBLE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate 'spotify-2023.csv'. Please place it in the 'data-music/' directory."
    )


def load_data(data_path=None, **kwargs):
    """Loads and preprocesses tabular audio features for neural network training."""
    csv_file = find_dataset_file(data_path)

    # 1. Load CSV data with compatible latin-1 encoding
    df = pd.read_csv(csv_file, encoding="latin-1")

    # Select only required feature columns and eliminate missing records
    required_cols = NUMERIC_FEATURES + list(TEXT_FEATURES.keys()) + [TARGET]
    df = df[required_cols].dropna()

    # 2. Map categorical text columns to numeric representations
    X = df[NUMERIC_FEATURES].copy()
    for col, mapping in TEXT_FEATURES.items():
        X[col] = df[col].map(mapping)

    # Filter invalid index mappings
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    df_clean = df.loc[valid_idx]

    # 3. Process target classification labels
    classes = sorted(df_clean[TARGET].unique().tolist())
    print("Detected classes:", classes)

    target_map = {name: idx for idx, name in enumerate(classes)}
    y = df_clean[TARGET].map(target_map)

    features = X.to_numpy(dtype="float32")
    labels = y.to_numpy(dtype="int32")

    print(f"Loaded dataset: {features.shape[0]} samples with {features.shape[1]} features")

    # Return standard tuple interface: features, labels, class names
    return features, labels, classes


if __name__ == "__main__":
    features, labels, classes = load_data()
    print("Features shape:", features.shape)
    print("Labels shape  :", labels.shape)
    print("Classes       :", classes)