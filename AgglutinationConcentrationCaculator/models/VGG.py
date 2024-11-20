import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_dir = r"C:\Users\XiaQi\Documents\Bristol\Individual_Project\OneDrive_2024-01-30\Augmented_Images4"
train_set = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)

test_dir = r"C:\Users\XiaQi\Documents\Bristol\Individual_Project\OneDrive_2024-01-30\Augmented_Images4"
test_set = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

# Load the pre-trained VGG16 model
model = models.vgg16(pretrained=True)
# Freeze
for param in model.features.parameters():
    param.requires_grad = False

# Replace the last classifier layer
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(train_set.classes))
model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train model
train_losses = []
for epoch in range(5):  # Iterate
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

# Test
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Caculate MAE, RMSE, R²
mae = mean_absolute_error(all_labels, all_preds)
rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
r_squared = r2_score(all_labels, all_preds)
print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
