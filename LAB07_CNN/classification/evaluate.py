import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


"""def plot_history(history, save_path):
    # Accuracy and loss curves — the main tool for spotting overfitting.

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")"""

def plot_hyperparameter_comparison(histories, save_path):

    """ Plot training and validation curves for multiple models in a 2x2 grid.
    histories: a dictionary containing training history for each model.
    save_path: file path to save the output image. """

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Loop through each model and plot its curves[cite: 5]
    for name, history in histories.items():
        # Get the list of epochs (1, 2, 3, ...)
        epochs_range = range(1, len(history.history["accuracy"]) + 1)
        
        # (a) Plot Training Accuracy[cite: 5]
        axes[0, 0].plot(epochs_range, history.history["accuracy"], label=name, linewidth=1.8)
        
        # (b) Plot Validation Accuracy[cite: 5]
        axes[0, 1].plot(epochs_range, history.history["val_accuracy"], label=name, linewidth=1.8)
        
        # (c) Plot Training Loss[cite: 5]
        axes[1, 0].plot(epochs_range, history.history["loss"], label=name, linewidth=1.8)
        
        # (d) Plot Validation Loss[cite: 5]
        axes[1, 1].plot(epochs_range, history.history["val_loss"], label=name, linewidth=1.8)

    # Set title and labels for (a) Training Accuracy[cite: 5]
    axes[0, 0].set_title("(a) Training accuracy of models", fontsize=11)
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    # Set title and labels for (b) Validation Accuracy[cite: 5]
    axes[0, 1].set_title("(b) Validation accuracy of models", fontsize=11)
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    # Set title and labels for (c) Training Loss[cite: 5]
    axes[1, 0].set_title("(c) Training loss of models", fontsize=11)
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    # Set title and labels for (d) Validation Loss[cite: 5]
    axes[1, 1].set_title("(d) Validation loss of models", fontsize=11)
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    # Adjust layout and save the figure
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved Comparison Plot: {save_path}")