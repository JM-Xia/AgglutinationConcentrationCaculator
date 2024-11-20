import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import re
import pickle
from collections import Counter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score



def save_checkpoint(model, optimizer=None, epoch=None, train_concentrations=None, val_concentrations=None, filename="model_checkpoint.pth"):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
    }

    if train_concentrations is not None:
        np.save('train_concentrations.npy', train_concentrations)
    if val_concentrations is not None:
        np.save('val_concentrations.npy', val_concentrations)

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filepath, model, optimizer=None):
    if not os.path.isfile(filepath):
        print(f"Checkpoint file '{filepath}' not found")
        return None

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state'])

    if 'optimizer_state' in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint.get('epoch')

    train_concentrations = np.load('train_concentrations.npy') if os.path.exists('train_concentrations.npy') else None
    val_concentrations = np.load('val_concentrations.npy') if os.path.exists('val_concentrations.npy') else None

    print(f"Checkpoint loaded from '{filepath}'")

    return epoch, train_concentrations, val_concentrations
img_dir = 'C:/Users/XiaQi/Documents/Bristol/Individual_Project/OneDrive_2024-01-30/Augmented_Images'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ConcentrationDataset(img_dir=img_dir, transform=transform, is_train=True)
val_dataset = ConcentrationDataset(img_dir=img_dir, transform=transform, is_train=False)

# Create DataLoader for train and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Get concentrations
train_concentrations = train_dataset.get_all_concentrations()
val_concentrations = val_dataset.get_all_concentrations()

# Find unique concentrations and print them
unique_train_concentrations = np.unique(train_concentrations)
unique_val_concentrations = np.unique(val_concentrations)

print(f"Unique concentrations in the training set: {unique_train_concentrations}")
print(f"Unique concentrations in the validation set: {unique_val_concentrations}")

# To inspect if there's a discrepancy in your plot, you can also print the actual counts:

train_concentration_counts = Counter(train_concentrations)
val_concentration_counts = Counter(val_concentrations)

print(f"Concentration counts in the training set: {train_concentration_counts}")
print(f"Concentration counts in the validation set: {val_concentration_counts}")




current_epoch = 0
model = SimpleCNN(dropout_rate=0.5)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#current_epoch, train_concentrations, val_concentrations = load_checkpoint(filepath="model_checkpoint.pth", model=model, optimizer=optimizer)


num_epochs = 50

train_losses = []  # 用于记录每个epoch的训练损失
test_losses = []  # 用于记录每个epoch的测试损失
n_splits = min(3, len(train_dataset))  # 确保分割数不超过样本数
if len(train_dataset) > 0:
    n_splits = min(3, len(train_dataset))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

else:
    raise RuntimeError("The training dataset is empty. Check the data loading and preprocessing steps.")

# 用于存储所有折叠的训练和验证损失
all_train_losses = []
all_val_losses = []
# 初始化最低验证损失为无穷大
best_val_loss = float('inf')
best_model_state = None
# 交叉验证循环
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
    model = SimpleCNN()
    print(f'Fold {fold + 1}/{n_splits}')

    # 创建训练和验证的 DataLoader
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_subsampler)

    # 初始化模型和优化器
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 用于存储当前折叠的训练和验证损失
    fold_train_losses = []
    fold_val_losses = []

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        current_epoch = epoch + 1
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            if inputs is None or targets is None:
                continue
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        save_checkpoint(model=model, optimizer=optimizer, epoch=current_epoch,
                        train_concentrations=train_concentrations,
                        val_concentrations=val_concentrations,
                        filename=f"model_checkpoint_epoch_{current_epoch}.pth")
      # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.float()
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # 计算当前epoch的平均训练和验证损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        fold_train_losses.append(avg_train_loss)
        fold_val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # 更新最低验证损失
            best_model_state = model.state_dict()  # 保存最佳模型状态
            print(f"New best model found at epoch {epoch + 1}, Fold {fold + 1} with Val Loss: {best_val_loss:.4f}")

        # save best
        torch.save(best_model_state, 'best_model.pth')

        model.load_state_dict(best_model_state)
        # 打印当前epoch的训练和验证损失
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # 将当前折叠的损失添加到全部损失中
    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)

# Save NumPy arrays to disk
np.save('train_concentrations.npy', train_concentrations)
np.save('val_concentrations.npy', val_concentrations)

save_checkpoint(model=model, optimizer=optimizer, epoch=current_epoch,
                train_concentrations=train_concentrations,
                val_concentrations=val_concentrations,
                filename="model_checkpoint_final.pth")

# 打印每一折最后一个epoch的平均训练和验证损失
for fold in range(n_splits):
    print(f"Fold {fold + 1}, Train Loss: {all_train_losses[fold][-1]:.4f}, Val Loss: {all_val_losses[fold][-1]:.4f}")
# Calculate the mean of the final losses across all folds
mean_train_loss = np.mean([losses[-1] for losses in all_train_losses])
mean_val_loss = np.mean([losses[-1] for losses in all_val_losses])

print(f"Mean Train Loss across folds: {mean_train_loss:.4f}")
print(f"Mean Val Loss across folds: {mean_val_loss:.4f}")


# Optionally, plot the average losses across epochs for all folds
plt.figure(figsize=(12, 6))
plt.plot(np.mean(all_train_losses, axis=0), label='Average Train Loss Across Folds')
plt.plot(np.mean(all_val_losses, axis=0), label='Average Val Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Across Folds (Log Scale)')
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.legend()
plt.show()

model.eval()  # Set the model to evaluation mode

actuals = []
predictions = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.float(), targets.float()
        outputs = model(inputs)

        actuals.extend(targets.cpu().numpy())
        predictions.extend(outputs.cpu().numpy())

# Convert lists to numpy arrays for plotting and analysis
actuals_array = np.array(actuals)
predictions_array = np.array(predictions)

mae = mean_absolute_error(actuals_array, predictions_array)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# RMSE Calculation
rmse = np.sqrt(mean_squared_error(actuals_array, predictions_array))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# R² Calculation
r_squared = r2_score(actuals_array, predictions_array)
print(f"R-squared (R²): {r_squared:.4f}")

log_actuals = [0 if value == 0 else np.log10(value) for value in actuals]
log_predictions = [0 if value == 0 else np.log10(value) for value in predictions]

# Convert the unique_concentrations back to floats, sort them, and then convert back to strings
unique_concentrations = np.unique(actuals_array)
sorted_concentrations = sorted(unique_concentrations, key=lambda x: float(x))
formatted_concentrations = [f"{10**np.floor(np.log10(float(c))) :.0e}" if float(c) != 0 else "0" for c in sorted_concentrations]

# Sorted_concentrations should be in the correct order
print(sorted_concentrations)

errors = actuals_array - log_predictions
errors = np.array(errors).flatten()
plt.hist(errors, bins=50, alpha=0.6, color='g')

# Before the plotting section, create a dictionary to store the actual concentration values
concentration_to_actuals = {}

# Gather actual concentration values for each unique concentration
for actual, predicted, conc in zip(actuals_array, predictions_array, val_concentrations):
    if conc not in concentration_to_actuals:
        concentration_to_actuals[conc] = {'actuals': [], 'predictions': []}
    concentration_to_actuals[conc]['actuals'].append(actual)
    concentration_to_actuals[conc]['predictions'].append(predicted)

# Verify the actual values and predictions are correct
print(f"Actuals: {actuals_array}")
print(f"Predictions: {predictions_array}")

# Use the actual log-transformed data for determining the plot range
min_log_value = min(np.min(log_actuals), np.min(log_predictions))
max_log_value = max(np.max(log_actuals), np.max(log_predictions))

# Print to debug
print("Min log value:", min_log_value)
print("Max log value:", max_log_value)


# Plotting each concentration's actual vs. predicted values
plt.figure(figsize=(10, 6))

plt.scatter(log_actuals, log_predictions, alpha=0.5)
plt.plot([min_log_value, max_log_value], [min_log_value, max_log_value], 'k--', lw=2)
plt.xlabel('Log of Actual Concentration')
plt.ylabel('Log of Predicted Concentration')
plt.title('Log of Actual vs. Log of Predicted Concentrations')

plt.legend()
plt.show()

# Continue using the errors as calculated
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.6, color='g')
plt.xlabel('Prediction Error (Exponent)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors (Exponent)')

# Display standard deviation on the plot
std_error = np.std(errors)
plt.figtext(0.15, 0.85, f'Std Dev of Errors: {std_error:.2f}', fontsize=12)
plt.show()
'''
print(f"Length of val_concentrations: {len(val_concentrations)}")
print(f"Length of actuals_array: {len(actuals_array)}")
print(f"Length of predictions_array: {len(predictions_array)}")
'''



def main(args):
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
    all_train_losses = []  # 存储每个epoch的训练损失
    all_val_losses = []  # 存储每个epoch的验证损失
    all_actuals = []  # 存储所有epoch的真实标签
    all_predictions = []  # 存储所有epoch的预测结果

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Starting fold {fold + 1}')
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        model = select_model(args.model, train_loader, val_loader, args)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss = float('inf')  # 初始化为无穷大


        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))  # 确保targets的维度与outputs匹配
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            all_train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            actuals = []  # 重置这个epoch的真实标签和预测结果
            predictions = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.float(), targets.float()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))  # 确保targets的维度与outputs匹配
                    val_loss += loss.item()
                    actuals.extend(targets.cpu().numpy())
                    predictions.extend(outputs.squeeze(1).cpu().numpy())  # 使用squeeze()确保维度匹配
            avg_val_loss = val_loss / len(val_loader)
            all_val_losses.append(avg_val_loss)
            all_actuals.extend(actuals)
            all_predictions.extend(predictions)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')

            # 打印当前epoch的损失信息
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 绘制损失曲线和预测分析图
    plot_losses(all_train_losses, all_val_losses)
    plot_prediction_analysis(np.array(all_actuals), np.array(all_predictions))
    plot_prediction_errors(np.array(all_actuals), np.array(all_predictions))

    # 假设 all_actuals 和 all_predictions 分别是存储所有实际值和预测值的列表
    actuals_array = np.array(all_actuals)
    predictions_array = np.array(all_predictions)

    # 计算 MAE
    mae = mean_absolute_error(actuals_array, predictions_array)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(actuals_array, predictions_array))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # 计算 R^2
    r_squared = r2_score(actuals_array, predictions_array)
    print(f"R-squared (R²): {r_squared:.4f}")