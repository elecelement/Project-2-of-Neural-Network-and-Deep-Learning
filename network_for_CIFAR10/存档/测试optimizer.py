import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 设置随机种子保证可重复性
torch.manual_seed(64)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
batch_size = 256
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义模型
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
        x = self.classifier(x)
        return x


# 训练和评估函数
def train_and_evaluate(optimizer_name, optimizer_params, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training with {optimizer_name} ===")

    # 初始化模型
    net = ResNetLikeNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 选择优化器
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(), **optimizer_params)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(net.parameters(), **optimizer_params)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), **optimizer_params)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), **optimizer_params)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), **optimizer_params)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=running_loss / (loop.n + 1), acc=100. * correct / total)

        scheduler.step()

    # 评估
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"{optimizer_name} test accuracy: {accuracy:.2f}%")
    return accuracy


# 主程序
if __name__ == '__main__':
    # 定义要测试的优化器及其参数
    optimizers = {
        'Adam': {'lr': 0.001},
        'AdamW': {'lr': 0.001, 'weight_decay': 0.01},
        'SGD': {'lr': 0.1, 'momentum': 0.9, 'nesterov': True},
        'RMSprop': {'lr': 0.001, 'alpha': 0.99},
        'Adagrad': {'lr': 0.01},
        'Adadelta': {'lr': 1.0}
    }

    results = {}
    for opt_name, params in optimizers.items():
        acc = train_and_evaluate(opt_name, params, epochs=25)
        results[opt_name] = acc

    # 打印最终结果比较
    print("\n=== Final Comparison ===")
    for opt_name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{opt_name}: {acc:.2f}%")