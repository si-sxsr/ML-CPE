import json
import os
from tensorflow import keras
from tensorflow.keras import layers

# Standard VGG-like configuration (deeper variant)
VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3)]

# Lightweight VGG configuration for faster CPU training and smaller datasets
VGG_SMALL = [(32, 2), (64, 2), (128, 2)]


def vgg_block(model, filters, n_conv):
    """VGG Block: n_conv x (Conv3x3 -> ReLU) followed by MaxPooling2D to halve spatial resolution."""
    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(2))


def build_model(input_shape, num_classes, blocks=VGG_SMALL):
    """Constructs the sequential CNN architecture with preprocessing and regularization."""
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # Rescale pixel intensities from [0, 255] to [0.0, 1.0]
    model.add(layers.Rescaling(1.0 / 255))

    # Spatial augmentations active only during training
    # Note: RandomFlip('horizontal') is omitted to avoid flipping hand orientations
    model.add(layers.RandomRotation(0.08))
    model.add(layers.RandomZoom(0.08))
    model.add(layers.RandomTranslation(0.05, 0.05))

    # Feature extraction via convolutional blocks
    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    # Classification head: Flatten preserves relative spatial structure of hand fingers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))

    # Output layer: sigmoid for binary classification, softmax for multiclass
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    # Compile with Adam optimizer using a stable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=20, batch_size=32, blocks=VGG_SMALL):
    """Trains the model with early stopping and learning rate scheduling."""
    model = build_model(X_train.shape[1:], num_classes, blocks)
    model.summary()

    callbacks = [
        # Halt training when validation loss stops improving to mitigate overfitting
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        # Reduce learning rate when loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist the trained model weights and training history metrics
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)
        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, history


def predict_model(model, X_test):
    """Generates predicted class indices for the test dataset."""
    probabilities = model.predict(X_test, verbose=0)

    # Binary classification thresholding
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    # Multiclass argmax selection
    return probabilities.argmax(axis=1)
    model = build_model(X_train.shape[1:], num_classes, blocks)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)