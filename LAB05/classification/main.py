import json
from pathlib import Path
import joblib
import numpy as np

from data_loader import load_data
from evaluate import evaluate_model
from preprocessing import to_features
from split_data import split_dataset
from svm_model import predict_svm, train_svm

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TEST_SIZE = 0.2


def main():
  print("--" * 30)
  print("SVM Audio Classification: Major vs Minor")
  print("--" * 30)

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  # Step 1: Load Dataset
  print("\n[Step 1] Loading dataset...")
  features, labels, classes = load_data()

  np.save(OUTPUT_DIR / "features.npy", features)
  np.save(OUTPUT_DIR / "labels.npy", labels)
  with open(OUTPUT_DIR / "classes.json", "w") as f:
    json.dump(classes, f)

  print("\nDataset loaded successfully.")
  print(f"Total samples : {len(features)}")
  print(f"Classes       : {classes}")

  # Step 2: Preprocessing
  print("\n[Step 2] Preprocessing features...")
  X = to_features(features)
  y = labels
  print(f"Feature shape: {X.shape}")

  # Step 3: Split Dataset
  print("\n[Step 3] Splitting dataset...")
  X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

  np.save(OUTPUT_DIR / "X_train.npy", X_train)
  np.save(OUTPUT_DIR / "X_test.npy", X_test)
  np.save(OUTPUT_DIR / "y_train.npy", y_train)
  np.save(OUTPUT_DIR / "y_test.npy", y_test)

  print(f"Training samples: {len(X_train)}")
  print(f"Testing samples : {len(X_test)}")

  # Step 4: Train and Compare 3 SVM Kernels
  kernels = ["linear", "poly", "rbf"]
  results = {}
  best_accuracy = -1.0
  best_kernel = None
  best_model = None
  best_scaler = None

  print("\n[Step 4] Training SVM with multiple kernels...")
  for kernel in kernels:
    print(f"\n--> Training with kernel: '{kernel}'")
    model, scaler = train_svm(X_train, y_train, kernel=kernel)
    predictions = predict_svm(model, scaler, X_test)

    acc = evaluate_model(y_test, predictions, classes)
    results[kernel] = acc

    if acc > best_accuracy:
      best_accuracy = acc
      best_kernel = kernel
      best_model = model
      best_scaler = scaler

  # Save best model and its artifacts
  joblib.dump(best_model, OUTPUT_DIR / "svm_model.pkl")
  joblib.dump(best_scaler, OUTPUT_DIR / "scaler.pkl")

  # Step 5: Summary and Evaluation for Best Kernel
  print("\n" + "=" * 60)
  print("Kernel Comparison Summary:")
  for k, score in results.items():
    print(f"  - Kernel '{k}': {score * 100:.2f}% accuracy")

  print(f"\nBest Kernel: '{best_kernel}' ({best_accuracy * 100:.2f}%)")
  print("=" * 60)

  print(f"\n[Step 5] Generating confusion matrix for best kernel ({best_kernel})...")
  best_predictions = predict_svm(best_model, best_scaler, X_test)
  evaluate_model(
      y_test,
      best_predictions,
      classes,
      save_path=str(OUTPUT_DIR / "confusion_matrix.png"),
  )


if __name__ == "__main__":
  main()