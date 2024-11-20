import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)  # Apply dropout after convolutions
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of the flattened features
        self._to_linear = None
        self._calculate_flat_size(torch.zeros(1, 3, 256, 256))


        # Dense layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def _calculate_flat_size(self, x):
        if self._to_linear is None:
            with torch.no_grad():
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = self.pool(F.relu(self.conv3(x)))
                x = self.pool(F.relu(self.conv4(x)))
                self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ResNetRegressor(nn.Module):
    def __init__(self):
        super(ResNetRegressor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class AttentionCNN(nn.Module):
    def __init__(self, input_size=(3, 256, 256)):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.se1 = SqueezeExcitation(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.se2 = SqueezeExcitation(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.se3 = SqueezeExcitation(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.se4 = SqueezeExcitation(256)
        self.pool = nn.MaxPool2d(2, 2)

        # Dynamically calculate the input feature size to the first fully connected layer
        self.fc_input_features = self._calc_fc1_input_features(input_size)
        self.fc1 = nn.Linear(self.fc_input_features, 256)
        self.fc2 = nn.Linear(256, 1)

    def _calc_fc1_input_features(self, input_size):
        # Create a dummy data input to simulate the forward path and determine input size
        with torch.no_grad():
            dummy_data = torch.zeros((1, *input_size))  # Create a dummy input
            x = self.pool(F.relu(self.conv1(dummy_data)))
            x = self.se1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.se2(x)
            x = self.pool(F.relu(self.conv3(x)))
            x = self.se3(x)
            x = self.pool(F.relu(self.conv4(x)))
            x = self.se4(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x.shape[1]  # Total number of output features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.se1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.se2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.se3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.se4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class TransferAttentionCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(TransferAttentionCNN, self).__init__()
        # Load a pre-trained ResNet and modify it
        base_model = models.resnet18(pretrained=True)
        layers = list(base_model.children())[:-2]
        self.features = nn.Sequential(*layers)

        self.se = SqueezeExcitation(512)

        # Additional layers for regression
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x