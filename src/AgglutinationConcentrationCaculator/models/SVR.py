from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

def extract_features_and_train_svr(train_dataset, val_dataset, args):
    # Initialize the feature extractor model
    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
    feature_extractor.eval()

    # DataLoader setup for both training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Function to extract features from a dataset
    def extract_features(loader):
        features = []
        labels = []
        for inputs, targets in loader:
            with torch.no_grad():
                outputs = feature_extractor(inputs).squeeze()
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
        features = np.vstack(features)  # Stacking for 2D array
        labels = np.concatenate(labels, axis=0)
        return features, labels

    # Extract features from both train and validation datasets
    X_train, y_train = extract_features(train_loader)
    X_val, y_val = extract_features(val_loader)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # SVR model training
    svr = SVR(C=1.0, epsilon=0.2)
    svr.fit(X_train_scaled, y_train)

    # Evaluation on the validation set
    predictions = svr.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r_squared = r2_score(y_val, predictions)

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R^2: {r_squared:.4f}")

    # Return the trained model, scaler, and metrics
    return svr, scaler, {'MAE': mae, 'RMSE': rmse, 'R^2': r_squared}

