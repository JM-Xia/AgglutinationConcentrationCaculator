import argparse
import os
import numpy as np
import torch
from torchvision import transforms
import joblib
from data.Concentration import ConcentrationDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.ResRF import ResNetRF

"""
Training Script for ResNet-RF Hybrid Model

This script trains a hybrid model combining ResNet18 and Random Forest for 
agglutination concentration prediction. It includes:

1. Dataset handling and k-fold cross validation
2. Model training and evaluation
3. Performance visualization
4. Model saving

The training process:
- Uses 3-fold cross validation
- Saves models after final fold
- Plots actual vs predicted concentrations

Usage:
   python main.py --img_dir PATH_TO_IMAGES --batch_size BATCH_SIZE
"""

def plot_prediction_analysis(actuals, predictions):
    log_actuals = [0 if value == 0 else np.log10(value) for value in actuals]
    log_predictions = [0 if value == 0 else np.log10(value) for value in predictions]
    plt.figure(figsize=(10, 6))
    plt.scatter(log_actuals, log_predictions, alpha=0.5)
    plt.plot([min(log_actuals), max(log_actuals)], [min(log_actuals), max(log_actuals)], 'k--', lw=2)
    plt.xlabel('Log of Actual Concentration')
    plt.ylabel('Log of Predicted Concentration')
    plt.title('Log of Actual vs Log of Predicted Concentrations')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNetRF model on concentration data.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--img_dir", type=str,
                        default=r"C:\Users\XiaQi\Documents\UW\bioen537\Augmented_Images",
                        help="Directory containing the images")
    return parser.parse_args()


def main(args):
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize dataset
    dataset = ConcentrationDataset(img_dir=args.img_dir, transform=transform)

    # Setup k-fold cross validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    all_actuals = []
    all_predictions = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nStarting fold {fold + 1}")

        # Create data loaders for this fold
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        # Train model
        feature_extractor, rf_model, scaler = ResNetRF(train_loader, val_loader)

        # Save models after last fold
        if fold == 2:
            print("\nSaving trained models...")
            if not os.path.exists('trained_models'):
                os.makedirs('trained_models')

            joblib.dump(rf_model, 'trained_models/rf_model.joblib')
            joblib.dump(scaler, 'trained_models/scaler.joblib')
            torch.save(feature_extractor.state_dict(), 'trained_models/feature_extractor.pth')
            print("Models saved successfully in 'trained_models' directory")

    # Visualize results
    if all_actuals and all_predictions:
        plot_prediction_analysis(all_actuals, all_predictions)


if __name__ == "__main__":
    args = parse_args()
    main(args)