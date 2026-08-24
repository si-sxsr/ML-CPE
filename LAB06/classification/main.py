import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

# Resolve dynamic paths relative to current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 32


def main():
    print("--" * 30)
    print("Neural Network Audio Classification: Major vs Minor")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Tabular Dataset
    print("\n[Step 1] Loading dataset...")
    features, labels, classes = load_data()

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total samples : {len(features)}")
    print(f"Classes       : {classes}")

    # Step 2: Preprocessing Features
    print("\n[Step 2] Preprocessing tabular features...")
    X = to_features(features)
    y = labels
    np.save(f"{OUTPUT_DIR}/features.npy", X)

    # Step 3: Split Dataset & Apply Scaling
    print("\n[Step 3] Splitting dataset and scaling features...")
    from preprocessing import scale_features
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_s)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val_s)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test_s)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    # Step 4: Compare Neural Network Configurations & Epochs
    experiment_configs = [
        {"name": "Config 1 (1 Layer [32], 30 Epochs)", "hidden_units": [32], "epochs": 30},
        {"name": "Config 2 (2 Layers [64, 32], 30 Epochs)", "hidden_units": [64, 32], "epochs": 30},
        {"name": "Config 3 (2 Layers [64, 32], 60 Epochs)", "hidden_units": [64, 32], "epochs": 60},
        {"name": "Config 4 (3 Layers [128, 64, 32], 60 Epochs)", "hidden_units": [128, 64, 32], "epochs": 60},
    ]

    best_acc = -1.0
    best_model = None
    best_history = None
    best_config_name = ""

    print("\n[Step 4] Training and comparing Neural Network configurations...")
    for cfg in experiment_configs:
        print(f"\n--> Running: {cfg['name']}")
        model, history = train_model(
            X_train_s, y_train, X_val_s, y_val, len(classes),
            OUTPUT_DIR, cfg["epochs"], BATCH_SIZE, hidden_units=cfg["hidden_units"]
        )

        preds = predict_model(model, X_test_s)
        acc = evaluate_model(y_test, preds, classes)
        print(f"Result for {cfg['name']} -> Test Accuracy: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_history = history
            best_config_name = cfg["name"]

    print("\n" + "=" * 60)
    print(f"Best Configuration: {best_config_name} ({best_acc * 100:.2f}%)")
    print("=" * 60)

    # Save artifacts for the best-performing model
    best_model.save(f"{OUTPUT_DIR}/nn_model.keras")
    with open(f"{OUTPUT_DIR}/history.json", "w") as f:
        json.dump(best_history.history, f)

    # Step 5: Final Evaluation & Visualizations
    print(f"\n[Step 5] Generating evaluation artifacts for best model ({best_config_name})...")
    best_predictions = predict_model(best_model, X_test_s)
    evaluate_model(
        y_test, best_predictions, classes,
        save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
    )
    plot_history(best_history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()