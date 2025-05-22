import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torchcam.methods import CAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import CIFAR10

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)


# Define the network architecture
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.Swish = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.Swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.Swish(out)
        return out


class ResNetLikeNet(nn.Module):
    def __init__(self, dropout_fc=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Swish()
        )
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            Swish(),
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


# Load test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Visualization functions with device handling
def visualize_filters(model, layer_name, device):
    model.eval()
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            filters = module.weight.data.cpu().numpy()  # Always move to CPU for visualization
            num_filters = min(16, filters.shape[0])

            fig, axes = plt.subplots(1, num_filters, figsize=(15, 2))
            fig.suptitle(f'Filters in {layer_name}', y=1.05)

            for i in range(num_filters):
                if filters.shape[1] == 3:
                    filter_img = filters[i].mean(axis=0)
                else:
                    filter_img = filters[i, 0]
                axes[i].imshow(filter_img, cmap='viridis')
                axes[i].axis('off')

            plt.tight_layout()
            filename = f'visualizations/filters_{layer_name.replace(".", "_")}.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"Saved filter visualization: {filename}")
            break


def visualize_feature_maps(model, image, layer_names, device):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = []
    for name, layer in model.named_modules():
        if name in layer_names:
            hooks.append(layer.register_forward_hook(get_activation(name)))

    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))  # Ensure input is on correct device

    for hook in hooks:
        hook.remove()

    for name in layer_names:
        act = activations[name].squeeze().cpu()  # Move to CPU for visualization
        num_feats = act.shape[0]
        num_cols = 8
        num_rows = (num_feats + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
        fig.suptitle(f'Feature maps: {name}', y=1.02)

        for i in range(num_feats):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].imshow(act[i], cmap='viridis')
            axes[row, col].axis('off')

        for i in range(num_feats, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        filename = f'visualizations/feature_maps_{name.replace(".", "_")}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved feature maps: {filename}")


def unnormalize(img_tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """将标准化后的图像还原到原始像素范围"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean


def visualize_cam(model, testset, class_names, device):
    from torchcam.methods import GradCAM
    from torchvision.transforms.functional import to_pil_image

    target_layer = 'layer4.2'  # 请确认此处层名是否正确
    cam_extractor = GradCAM(model, target_layer)

    model.eval()
    samples = [testset[i] for i in np.random.choice(len(testset), 3, replace=False)]

    plt.figure(figsize=(15, 12))
    for i, (img, label) in enumerate(samples):
        input_tensor = img.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # 提取 CAM
        cams = cam_extractor(pred_class, output)
        cam_map = cams[0].cpu()
        cam_map -= cam_map.min()
        cam_map /= cam_map.max() + 1e-8  # 避免除0

        # 去标准化图像用于显示
        img_unnorm = unnormalize(img.cpu()).clamp(0, 1)

        # 创建叠加图
        overlay = overlay_mask(to_pil_image(img_unnorm), to_pil_image(cam_map, mode='F'), alpha=0.5)

        # 显示未标准化原图
        plt.subplot(3, 4, i * 4 + 1)
        plt.imshow(to_pil_image(img_unnorm))
        plt.title(f"Unnormalized\nTrue: {class_names[label]}")
        plt.axis('off')

        # 标准化图像（用于对比）
        plt.subplot(3, 4, i * 4 + 2)
        plt.imshow(to_pil_image(img.cpu()))
        plt.title("Standardized")
        plt.axis('off')

        # 显示 CAM 热力图
        plt.subplot(3, 4, i * 4 + 3)
        plt.imshow(cam_map.numpy(), cmap='jet')
        plt.title("Activation Map")
        plt.axis('off')

        # 显示叠加图
        plt.subplot(3, 4, i * 4 + 4)
        plt.imshow(overlay)
        plt.title(f"Predicted: {class_names[pred_class]}")
        plt.axis('off')

    plt.tight_layout()
    filename = 'visualizations/class_activation_maps_full.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved CAM visualization with unnormalized images: {filename}")




# Main execution
if __name__ == '__main__':
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # Initialize model and move to device
    net = ResNetLikeNet().to(device)

    # Load saved model
    model_path = 'cifar_net.pth'
    if os.path.exists(model_path):
        # Load with map_location to handle device placement
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train and save the model first.")

    # Get a sample image for visualization
    sample_img, _ = testset[0]

    # Generate all visualizations
    print("\nGenerating visualizations...")
    visualize_filters(net, 'layer1.0', device)
    visualize_feature_maps(net, sample_img, ['layer1', 'layer2.0', 'layer3.0', 'layer4.0'], device)
    visualize_cam(net, testset, classes, device)


    print("\nAll visualizations saved to the 'visualizations' directory.")

