"""
Read CSV (spotify-2023.csv)
convert text to number
make Scaling for KNN
split data: train / validation / test
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to data-music
CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data-music"
    / "spotify-2023.csv"
)

# TARGET is mode (Major / Minor) 
TARGET = "mode"

# Audio features => Features
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

# convert Key to number
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


def load_data(test_size=0.2, seed=42):
  # Step 1 : Read CSV (encoding)
  df = pd.read_csv(CSV_PATH, encoding="latin-1")

  required_cols = NUMERIC_FEATURES + list(TEXT_FEATURES.keys()) + [TARGET]
  df = df[required_cols].dropna()

  # Step 2 : Convert features to numeric
  X = df[NUMERIC_FEATURES].copy()
  for col, mapping in TEXT_FEATURES.items():
    X[col] = df[col].map(mapping)

  valid_idx = X.dropna().index
  X = X.loc[valid_idx]
  df_clean = df.loc[valid_idx]

  class_names = sorted(df_clean[TARGET].unique())
  y = df_clean[TARGET].map({name: i for i, name in enumerate(class_names)})

  X = X.to_numpy(dtype="float32")
  y = y.to_numpy(dtype="int32")

  # Step 3 : Split data : Train 60% / Validation 20% / Test 20%
  X_temp, X_test, y_temp, y_test = train_test_split(
      X, y, test_size=test_size, random_state=seed, stratify=y
  )

  X_train, X_val, y_train, y_val = train_test_split(
      X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
  )

  # Step 4 : Scaling
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train).astype("float32")
  X_val = scaler.transform(X_val).astype("float32")
  X_test = scaler.transform(X_test).astype("float32")

  return {
      "X_train": X_train,
      "y_train": y_train,
      "X_val": X_val,
      "y_val": y_val,
      "X_test": X_test,
      "y_test": y_test,
      "class_names": class_names,
      "feature_names": NUMERIC_FEATURES + list(TEXT_FEATURES.keys()),
      "n_rows": len(df_clean),
  }


if __name__ == "__main__":
  data = load_data()
  print("train :", data["X_train"].shape)
  print("val   :", data["X_val"].shape)
  print("test  :", data["X_test"].shape)
  print("คลาส  :", data["class_names"])
