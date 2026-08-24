import numpy as np


def to_features(data):
  """Ensures feature matrix is a 2D numpy array with float32 precision."""
  if not isinstance(data, np.ndarray):
    data = np.array(data)

  features = data.reshape(len(data), -1).astype(np.float32)
  return features


def preprocess_data(data):
  """Pass-through wrapper to maintain pipeline compatibility."""
  return to_features(data)