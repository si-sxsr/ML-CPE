"""
Test the trained Neural Network model on random Spotify audio feature samples.
Generates predictions, displays confidence metrics, and saves a summary table.
"""

import json
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 8


def test_nn(n_samples=N_SAMPLES):
    # Load trained model and test split data
    model = keras.models.load_model(os.path.join(OUTPUT_DIR, "nn_model.keras"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    with open(os.path.join(OUTPUT_DIR, "classes.json")) as f:
        classes = json.load(f)

    # Randomly select sample instances from test set
    index = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # Predict probabilities
    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    print("-" * 60)
    print("Neural Network Random Sample Evaluation:")
    print("-" * 60)

    sample_results = []
    for i in range(len(index)):
        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        conf_pct = f"{confidence[i] * 100:.1f}%"

        sample_results.append({
            "Sample": i + 1,
            "Predicted": pred,
            "True": true,
            "Confidence": conf_pct,
            "Result": "CORRECT" if correct else "WRONG"
        })

        print(
            f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
            f"Conf: {conf_pct:>6}  {'OK' if correct else 'WRONG'}"
        )

    correct_total = int((predictions == y_sample).sum())
    print("-" * 60)
    print(f"Correct: {correct_total}/{len(index)} ({(correct_total/len(index))*100:.1f}%)")

    # Generate summary image table
    df_results = pd.DataFrame(sample_results)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_results.values,
        colLabels=df_results.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.3)

    fig.suptitle(f"Prediction Summary: {correct_total}/{len(index)} Correct", fontsize=12)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "prediction_sample.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_nn()