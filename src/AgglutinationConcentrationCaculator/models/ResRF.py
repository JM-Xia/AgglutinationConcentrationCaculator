import torch
import torch.nn as nn
from torchvision import models
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def ResNetRF(train_loader, val_loader=None, n_estimators=100, random_state=42):
    """
    ResNet + Random Forest hybrid model for agglutination concentration prediction.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        n_estimators: Number of trees in Random Forest
        random_state: Random state for reproducibility

    Returns:
        feature_extractor: Trained ResNet model
        rf: Trained Random Forest model
        scaler: Fitted StandardScaler
    """

    # Initialize ResNet feature extractor
    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor.fc = nn.Identity()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)

    # Extract features using ResNet
    def extract_features(loader):
        features = []
        labels = []
        feature_extractor.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = feature_extractor(inputs)
                features.append(outputs.cpu().numpy())
                labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(labels)

    # Get training features
    X_train, y_train = extract_features(train_loader)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train_scaled, y_train)

    # Validate if validation loader provided
    if val_loader is not None:
        X_val, y_val = extract_features(val_loader)
        X_val_scaled = scaler.transform(X_val)

        # Calculate metrics
        train_pred = rf.predict(X_train_scaled)
        val_pred = rf.predict(X_val_scaled)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
        print(f"Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}")

    return feature_extractor, rf, scaler