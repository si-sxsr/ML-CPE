from pathlib import Path
import numpy as np
import pandas as pd

# Default path to the Spotify 2023 dataset
CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data-music"
    / "spotify-2023.csv"
)
TARGET = "mode"

# Continuous audio features
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

# Categorical mapping for musical keys
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


def load_data(data_path=None, **kwargs):
  """Loads and cleans dataset, converting categorical features to numeric representations."""
  csv_file = Path(data_path) if data_path else CSV_PATH

  # 1. Read CSV file with compatible encoding
  df = pd.read_csv(csv_file, encoding="latin-1")

  # Filter required columns and drop missing entries
  required_cols = NUMERIC_FEATURES + list(TEXT_FEATURES.keys()) + [TARGET]
  df = df[required_cols].dropna()

  # 2. Map categorical text features to numerical values
  X = df[NUMERIC_FEATURES].copy()
  for col, mapping in TEXT_FEATURES.items():
    X[col] = df[col].map(mapping)

  # Filter out unmapped rows
  valid_idx = X.dropna().index
  X = X.loc[valid_idx]
  df_clean = df.loc[valid_idx]

  # 3. Process target labels and identify distinct classes
  classes = sorted(df_clean[TARGET].unique().tolist())
  print("Detected classes:", classes)

  target_map = {name: idx for idx, name in enumerate(classes)}
  y = df_clean[TARGET].map(target_map)

  features = X.to_numpy(dtype="float32")
  labels = y.to_numpy(dtype="int32")

  print(
      f"Loaded dataset: {features.shape[0]} samples with {features.shape[1]}"
      " features"
  )

  # Maintain compatibility with the original tuple output structure
  return features, labels, classes


if __name__ == "__main__":
  features, labels, classes = load_data()
  print("Features shape:", features.shape)
  print("Labels shape  :", labels.shape)
  print("Classes       :", classes)