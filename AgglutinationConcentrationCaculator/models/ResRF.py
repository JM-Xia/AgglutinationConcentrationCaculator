from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
import torch
import numpy as np
def train_random_forest_model(train_loader, val_loader, n_estimators=100, random_state=42):

    # Feature extraction might be necessary depending on the input type
    def extract_features_and_targets(loader):
        features = []
        targets = []
        for inputs, target in loader:

            features.append(inputs.numpy())  # Convert torch.Tensor to numpy array
            targets.append(target.numpy())
        return np.concatenate(features), np.concatenate(targets)

    # Extract features and targets
    X_train, y_train = extract_features_and_targets(train_loader)
    X_val, y_val = extract_features_and_targets(val_loader)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Training RandomForest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train_scaled, y_train)

    # Predicting and evaluating
    train_predictions = rf.predict(X_train_scaled)
    val_predictions = rf.predict(X_val_scaled)
    train_mse = mean_squared_error(y_train, train_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    val_r2 = r2_score(y_val, val_predictions)

    print(f"Train MSE: {train_mse:.4f}, Train R^2: {train_r2:.4f}")
    print(f"Validation MSE: {val_mse:.4f}, Validation R^2: {val_r2:.4f}")

    # Return the trained model and scaler for potential use
    return rf, scaler

from sklearn.ensemble import RandomForestRegressor

def train_random_forest(features, targets):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features, targets)
    return rf

def extract_features(dataset, model, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []

    model.eval()  # Ensure the model is in eval mode for feature extraction
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            outputs = torch.flatten(outputs, start_dim=1)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels
