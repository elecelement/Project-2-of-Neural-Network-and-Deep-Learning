import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(64)
random.seed(64)
np.random.seed(64)

# Data loading and transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_aug)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# Center Loss (for feature regularization)
class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, x, labels):
        batch_size = x.size(0)
        centers_batch = self.centers[labels]
        dist = (x - centers_batch).pow(2).sum(dim=1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean()
        return loss


# Your model definition remains the same
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetLikeNet(nn.Module):
    def __init__(self, dropout_fc=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Store features for center loss
        self.features = None

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, 10)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks=2):
        layers = []
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(ResidualBlock(in_channels, out_channels, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # Store features for center loss
        self.features = x

        x = self.classifier(x)
        return x


def train_and_evaluate(loss_type='cross_entropy', label_smoothing=0.1, weight_decay=0,
                       focal_alpha=1, focal_gamma=2, center_loss_weight=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining with {loss_type} loss...")

    # Initialize model
    net = ResNetLikeNet()
    net.to(device)

    # Initialize loss functions
    if loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'combined':
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        center_criterion = CenterLoss(device=device)
    else:
        raise ValueError("Invalid loss type")

    # Initialize optimizer with weight decay
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training loop
    for epoch in range(25):
        running_loss = 0.0
        loop = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}")

        for i, data in loop:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            if loss_type == 'combined':
                ce_loss = criterion(outputs, labels)
                center_loss = center_criterion(net.features, labels)
                loss = ce_loss + center_loss_weight * center_loss
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (i + 1))

        scheduler.step()

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy with {loss_type} loss: {accuracy:.2f}%')
    return accuracy


# Run experiments
results = {}

# 1. Baseline - CrossEntropy with label smoothing (0.1)
results['baseline'] = train_and_evaluate(loss_type='cross_entropy', label_smoothing=0.1)

# 2. Focal Loss
results['focal'] = train_and_evaluate(loss_type='focal', focal_alpha=1, focal_gamma=2)

# 3. More aggressive label smoothing (0.2)
results['smoothing_0.2'] = train_and_evaluate(loss_type='cross_entropy', label_smoothing=0.2)

# 4. CrossEntropy with weight decay (L2 regularization)
results['weight_decay'] = train_and_evaluate(loss_type='cross_entropy', label_smoothing=0.1, weight_decay=1e-4)

# 5. Combined loss (CrossEntropy + Center Loss)
results['combined'] = train_and_evaluate(loss_type='combined', label_smoothing=0.1, center_loss_weight=0.1)

# Print results summary
print("\n=== Results Summary ===")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")