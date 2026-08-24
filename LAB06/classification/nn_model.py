import json
import os
import tensorflow as tf
import keras
from keras import layers


def build_model(input_dim, num_classes=2, hidden_units=[32, 16], dropout_rate=0.4):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        # Removed BatchNormalization to stabilize small tabular dataset
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    output_units = 1 if num_classes == 2 else num_classes
    output_activation = "sigmoid" if num_classes == 2 else "softmax"
    loss_function = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"

    model.add(layers.Dense(output_units, activation=output_activation))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_function,
        metrics=["accuracy"],
    )
    return model


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    output_dir=None,
    epochs=30,
    batch_size=32,
    hidden_units=[64, 32],
):
    """Compiles, trains and returns the Neural Network model along with training history."""
    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_classes, hidden_units=hidden_units)

    # Callbacks for learning rate adaptation and overfitting prevention
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "nn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(
                {k: [float(v) for v in vs] for k, vs in history.history.items()},
                f,
            )

    return model, history


def predict_model(model, X_test):
    """Generates discrete class predictions from raw output probabilities."""
    probabilities = model.predict(X_test, verbose=0)

    # Single output unit thresholding for binary classification
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)