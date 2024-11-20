import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib
from data.Concentration import ConcentrationDataset
from models.CNN import SimpleCNN, CNN, ResNetRegressor, AttentionCNN, TransferAttentionCNN
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import Counter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_prediction_analysis(actuals, predictions):
    # Apply log transformation, handle zeros specifically
    log_actuals = [0 if value == 0 else np.log10(value) for value in actuals]
    log_predictions = [0 if value == 0 else np.log10(value) for value in predictions]
    # Assuming actuals and predictions are numpy arrays
    plt.figure(figsize=(10, 6))
    plt.scatter(log_actuals, log_predictions, alpha=0.5)
    plt.plot([min(log_actuals), max(log_actuals)], [min(log_actuals), max(log_actuals)], 'k--', lw=2)
    plt.xlabel('Log of Actual Concentration')
    plt.ylabel('Log of Predicted Concentration')
    plt.title('Log of Actual vs Log of Predicted Concentrations')
    plt.legend()
    plt.show()

def plot_prediction_errors(actuals, predictions):
    # Calculate errors
    errors = predictions - actuals

    # Plot histogram of prediction errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.6, color='g')
    plt.xlabel('Prediction Error (Exponent)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Exponent)')

    # Calculate and display standard deviation of errors
    std_error = np.std(errors)
    plt.figtext(0.15, 0.85, f'Std Dev of Errors: {std_error:.2f}', fontsize=12)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network on concentration data.")
    parser.add_argument("--model", type=str, default="RandomForest", choices=["SimpleCNN", "CNN",'ResNetRegressor','SVR','AttentionCNN','TransferAttentionCNN','VGG','RandomForest'],
                        help="The model to use for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for the optimizer")
    parser.add_argument("--img_dir", type=str,
                        default=r"C:\Users\XiaQi\Documents\Bristol\Individual_Project\OneDrive_2024-01-30\Augmented_Images5",
                        help="Directory containing the images")
    return parser.parse_args()

def select_model(model_name):
    if model_name == 'SimpleCNN':
        return (SimpleCNN(), 'deep_learning')
    elif model_name == 'CNN':
        return (CNN(), 'deep_learning')
    elif model_name == 'ResNetRegressor':
        return (ResNetRegressor(), 'deep_learning')
    elif model_name == 'AttentionCNN':
        return (AttentionCNN(), 'deep_learning')
    elif model_name == 'TransferAttentionCNN':
        return (TransferAttentionCNN(), 'deep_learning')
    elif model_name == 'VGG':
        model = models.vgg16(pretrained=True)  # load pre-trained model
        # Frozen
        for param in model.features.parameters():
            param.requires_grad = False
        # Get the number of input features of the original classifier (the layer before the last layer)
        num_features = model.classifier[6].in_features
        # Replace the last layer with a new linear layer
        model.classifier[6] = nn.Linear(num_features, 1)
        return (model, 'deep_learning')
    elif model_name == 'RandomForest':
        feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_extractor.fc = nn.Identity()
        return (feature_extractor, 'random_forest')
    else:
        print(f"No model found for {model_name}")
        return (None, None)
def train_deep_learning_model(model, train_loader, val_loader, args, all_train_losses, all_val_losses, all_actuals, all_predictions):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Calculate and record the average training loss
        avg_train_loss = train_loss / len(train_loader)
        all_train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        actuals = []
        predictions = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.float())
                val_loss += loss.item()
                actuals.extend(targets.cpu().numpy())
                predictions.extend(outputs.squeeze().cpu().numpy())

        # Calculate and record the average validation loss
        avg_val_loss = val_loss / len(val_loader)
        all_val_losses.append(avg_val_loss)
        all_actuals.extend(actuals)
        all_predictions.extend(predictions)

        # Print losses for the epoch
        print(f'Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}')


def evaluate_and_visualize(all_train_losses, all_val_losses, all_actuals, all_predictions):
    if not all_actuals or not all_predictions:
        print("No data was collected for evaluation. Check your validation loops.")
        return

    if not all_train_losses or not all_val_losses:
        print("Training or validation losses are missing. Check your training loop.")
        return

    # Convert lists to numpy arrays for evaluation
    actuals_array = np.array(all_actuals)
    predictions_array = np.array(all_predictions)

    # Calculate evaluation metrics
    mae = mean_absolute_error(actuals_array, predictions_array)
    rmse = np.sqrt(mean_squared_error(actuals_array, predictions_array))
    r_squared = r2_score(actuals_array, predictions_array)

    # Print evaluation metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r_squared:.4f}")

    # Plot training/validation losses and prediction accuracy
    #plot_losses(all_train_losses, all_val_losses)
    plot_prediction_analysis(actuals_array, predictions_array)
    plot_prediction_errors(actuals_array, predictions_array)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Initialize dataset and dataloader
    train_dataset = ConcentrationDataset(img_dir=args.img_dir, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Initialize validation dataset
    val_dataset = ConcentrationDataset(img_dir=args.img_dir, transform=transform, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Extract concentrations from datasets
    train_concentrations = [label for _, label in train_dataset]
    val_concentrations = [label for _, label in val_dataset]

    # Find unique concentrations and print them
    unique_train_concentrations = np.unique(train_concentrations)
    unique_val_concentrations = np.unique(val_concentrations)

    print(f"Unique concentrations in the training set: {unique_train_concentrations}")
    print(f"Unique concentrations in the validation set: {unique_val_concentrations}")

    # To inspect if there's a discrepancy in your plot, you can also print the actual counts
    train_concentration_counts = Counter(train_concentrations)
    val_concentration_counts = Counter(val_concentrations)

    print(f"Concentration counts in the training set: {train_concentration_counts}")
    print(f"Concentration counts in the validation set: {val_concentration_counts}")

    dataset = ConcentrationDataset(img_dir=args.img_dir, transform=transform)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    all_train_losses = []  # Store training losses for this fold
    all_val_losses = []  # Store validation losses for this fold
    all_actuals = []  # Store actual labels for evaluations
    all_predictions = []  # Store model predictions for evaluations
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Starting fold {fold + 1}")
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        model, model_type = select_model(args.model)
        if model is None:
            print(f"No model found for {model_type}. Skipping fold.")
            continue

        model.to(args.device)

        if model_type == 'deep_learning':
            train_deep_learning_model(model, train_loader, val_loader, args, all_train_losses, all_val_losses,
                                      all_actuals, all_predictions)
        elif model_type == 'random_forest':
            # Extract features
            print("Extracting features...")
            features = []
            labels = []
            model.eval()
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs = outputs.view(outputs.size(0), -1).cpu().numpy()
                    features.append(outputs)
                    labels.extend(targets.numpy())

            X_train = np.concatenate(features, axis=0)
            y_train = np.array(labels)

            # Extract validation features
            val_features = []
            val_labels = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs = outputs.view(outputs.size(0), -1).cpu().numpy()
                    val_features.append(outputs)
                    val_labels.extend(targets.numpy())

            X_val = np.concatenate(val_features, axis=0)
            y_val = np.array(val_labels)

            # Train Random Forest
            print("Training Random Forest...")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)

            # Get predictions
            predictions = rf.predict(X_val_scaled)

            # Collect results
            all_actuals.extend(y_val)
            all_predictions.extend(predictions)

            # Print metrics for this fold
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            print(f"Fold {fold + 1} - MSE: {mse:.4f}, R2: {r2:.4f}")
            # Save model
            if fold == 2:  # save model after the last folder finished
                print("\nSaving trained models...")
                if not os.path.exists('trained_models'):
                    os.makedirs('trained_models')

                # Save Random Forest model
                joblib.dump(rf, 'trained_models/rf_model.joblib')

                # Save scaler
                joblib.dump(scaler, 'trained_models/scaler.joblib')

                # Save feature extractor
                torch.save(model.state_dict(), 'trained_models/feature_extractor.pth')

                print("Models saved successfully in 'trained_models' directory")

        else:
            print(f"Unsupported model type: {model_type}. Skipping fold.")
            continue


    evaluate_and_visualize(all_train_losses, all_val_losses, all_actuals, all_predictions)


def __getitem__(self, index):
    # Loading data and target
    data, target = self.some_loading_function(index)

    # Convert to float
    data = data.float()
    target = target.float()

    return data, target


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0
    actuals = []
    predictions = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            total_loss += loss.item()
            actuals.extend(targets.cpu().numpy())
            predictions.extend(outputs.squeeze().cpu().numpy())

    # Calculate average validation loss
    avg_val_loss = total_loss / len(val_loader)

    # Optionally, you might want to return actuals and predictions if needed elsewhere
    return avg_val_loss, actuals, predictions

if __name__ == "__main__":
    args = parse_args()
    main(args)
