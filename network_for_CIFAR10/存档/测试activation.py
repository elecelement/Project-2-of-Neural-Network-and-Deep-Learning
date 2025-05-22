import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from tqdm import tqdm

torch.manual_seed(64)
random.seed(64)
np.random.seed(64)


# Define Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Define activation functions to try
ACTIVATIONS = {
    'ReLU': nn.ReLU(inplace=True),
    'LeakyReLU': nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'ELU': nn.ELU(alpha=1.0, inplace=True),
    'Swish': Swish()
}


def train_and_evaluate(activation_name, activation_fn):
    print(f"\nTraining with {activation_name} activation...")

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

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.activation = activation_fn
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=dropout_prob)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.activation(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.dropout(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.activation(out)
            return out

    class ResNetLikeNet(nn.Module):
        def __init__(self, dropout_fc=0.2):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                activation_fn
            )

            self.layer2 = self._make_layer(64, 128)
            self.layer3 = self._make_layer(128, 256)
            self.layer4 = self._make_layer(256, 512)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = nn.Sequential(
                nn.Linear(512, 512),
                activation_fn,
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
            x = self.classifier(x)
            return x

    net = ResNetLikeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    net.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(25):
        running_loss = 0.0
        loop = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}")
        for i, data in loop:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (i + 1))
        scheduler.step()

    print('Finished Training')

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
    print(f'Accuracy with {activation_name}: {accuracy:.2f}%')
    return accuracy


# Run experiments with different activations
results = {}
for name, activation in ACTIVATIONS.items():
    results[name] = train_and_evaluate(name, activation)

# Print summary of results
print("\nSummary of Results:")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}% accuracy")