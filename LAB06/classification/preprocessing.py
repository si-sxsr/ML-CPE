import numpy as np
from sklearn.preprocessing import StandardScaler


def to_features(data):
    """Ensures input data is formatted as a 2D float32 numpy array for MLP ingestion."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    features = data.reshape(len(data), -1).astype(np.float32)
    return features


def scale_features(X_train, X_val, X_test):
    """Standardizes features across Train, Validation, and Test splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler