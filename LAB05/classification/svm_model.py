from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, kernel="rbf", C=1.0, degree=3, gamma="scale"):
  """Trains an SVM classifier with standard feature scaling.

  Supports Linear, Polynomial, and RBF kernels.
  """
  # Use StandardScaler to standardize numerical audio features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)

  # Initialize Support Vector Classifier with specified kernel parameters
  model = SVC(
      kernel=kernel,
      C=C,
      degree=degree,
      gamma=gamma,
      random_state=42,
      cache_size=1000,
  )

  # Fit SVM model on scaled training data
  model.fit(X_train_scaled, y_train)

  return model, scaler


def predict_svm(model, scaler, X_test):
  """Transforms test features using fitted scaler and returns class predictions."""
  # Transform test features with training scaler parameters
  X_test_scaled = scaler.transform(X_test)

  # Generate predictions
  predictions = model.predict(X_test_scaled)

  return predictions