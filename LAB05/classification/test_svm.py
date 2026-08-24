import json
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
N_SAMPLES = 8


def test_svm(n_samples=N_SAMPLES):
  # Load saved model, scaler, and test split
  model = joblib.load(OUTPUT_DIR / "svm_model.pkl")
  scaler = joblib.load(OUTPUT_DIR / "scaler.pkl")
  X_test = np.load(OUTPUT_DIR / "X_test.npy")
  y_test = np.load(OUTPUT_DIR / "y_test.npy")

  with open(OUTPUT_DIR / "classes.json") as f:
    classes = json.load(f)

  # Pick random sample instances from test set
  indices = np.random.choice(
      len(X_test), min(n_samples, len(X_test)), replace=False
  )
  X_sample = X_test[indices]
  y_sample = y_test[indices]

  # Generate predictions using fitted scaler and SVM model
  X_sample_scaled = scaler.transform(X_sample)
  predictions = model.predict(X_sample_scaled)

  print("-" * 50)
  print("Random Sample Predictions (Spotify Audio Features):")
  print("-" * 50)

  sample_results = []
  for i in range(len(indices)):
    pred = classes[predictions[i]]
    true = classes[y_sample[i]]
    is_correct = predictions[i] == y_sample[i]

    sample_results.append({
        "Sample": i + 1,
        "True Label": true,
        "Predicted": pred,
        "Match": "CORRECT" if is_correct else "WRONG",
    })

    print(
        f"[{i + 1}] Predicted: {pred:<6} | True: {true:<6} |"
        f" {'OK' if is_correct else 'WRONG'}"
    )

  correct_total = int((predictions == y_sample).sum())
  accuracy_sample = (correct_total / len(indices)) * 100
  print("-" * 50)
  print(
      f"Sample Accuracy: {correct_total}/{len(indices)} ({accuracy_sample:.1f}%)"
  )

  # Export sample results table as an image
  df_results = pd.DataFrame(sample_results)
  fig, ax = plt.subplots(figsize=(6, 3))
  ax.axis("tight")
  ax.axis("off")
  table = ax.table(
      cellText=df_results.values,
      colLabels=df_results.columns,
      loc="center",
      cellLoc="center",
  )
  table.auto_set_font_size(False)
  table.set_fontsize(10)
  table.scale(1.2, 1.2)

  save_path = OUTPUT_DIR / "prediction_sample.png"
  fig.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved sample visualization to: {save_path}")


if __name__ == "__main__":
  test_svm()