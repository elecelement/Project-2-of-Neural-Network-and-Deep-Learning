import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
import matplotlib
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A, VGG_A_Dropout, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

matplotlib.use('TkAgg')

# Constants initialization
device_id = [0, 1, 2, 3]
num_workers = 4
batch_size = 128

# Path setup
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Device setup
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

# Data loaders
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)


# Accuracy function
def get_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Set random seeds
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Modified training function to return all curves
def train_model(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=20):
    model.to(device)
    learning_curve = []
    train_accuracy_curve = []
    val_accuracy_curve = []
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        epoch_losses = []
        epoch_grads = []

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            # Record loss and gradient
            epoch_losses.append(loss.item())
            grad = model.classifier[4].weight.grad.clone()
            epoch_grads.append(grad.cpu())

            optimizer.step()

        # Store epoch metrics
        losses_list.append(epoch_losses)
        grads.append(torch.mean(torch.stack(epoch_grads), dim=0))
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)

        learning_curve.append(np.mean(epoch_losses))
        train_accuracy_curve.append(train_acc)
        val_accuracy_curve.append(val_acc)


    return {
        'learning_curve': learning_curve,
        'train_accuracy': train_accuracy_curve,
        'val_accuracy': val_accuracy_curve,
        'losses_list': losses_list,
        'grads': grads
    }


# Main training loop for both models with different learning rates
def run_experiments():
    set_random_seeds(seed_value=2020, device=device)
    epochs_n = 20
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    criterion = nn.CrossEntropyLoss()

    # Store all results
    all_results = {
        'VGG_A_Dropout': {},
        'VGG_A_BatchNorm': {}
    }

    for lr in learning_rates:
        print(f"\n{'=' * 50}")
        print(f"Training with learning rate: {lr}")
        print(f"{'=' * 50}")

        # Train VGG_A_Dropout
        print("\nTraining VGG_A_Dropout...")
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        results = train_model(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs_n)
        all_results['VGG_A_Dropout'][lr] = results

        # Train VGG_A_BatchNorm
        print("\nTraining VGG_A_BatchNorm...")
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        results = train_model(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs_n)
        all_results['VGG_A_BatchNorm'][lr] = results

    return all_results


# Plot comparison curves
def plot_comparison_results(all_results):
    learning_rates = list(next(iter(all_results.values())).keys())
    # 修正：先将dict_values转换为list再取第一个元素
    first_model_results = list(all_results.values())[0]
    first_lr_results = list(first_model_results.values())[0]
    epochs_n = len(first_lr_results['learning_curve'])

    # Create figures with 2 rows and 2 columns (for 4 learning rates)
    plt.figure(figsize=(18, 12))

    # Plot accuracy comparison
    for i, lr in enumerate(learning_rates, 1):
        plt.subplot(2, 2, i)
        plt.title(f'Accuracy (LR: {lr})')

        # Plot VGG_A_Dropout curves
        dropout_data = all_results['VGG_A_Dropout'][lr]
        plt.plot(dropout_data['train_accuracy'], 'b--', label='VGG Train')
        plt.plot(dropout_data['val_accuracy'], 'b-', label='VGG Val')

        # Plot VGG_A_BatchNorm curves
        batchnorm_data = all_results['VGG_A_BatchNorm'][lr]
        plt.plot(batchnorm_data['train_accuracy'], 'r--', label='BatchNorm Train')
        plt.plot(batchnorm_data['val_accuracy'], 'r-', label='BatchNorm Val')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'accuracy_comparison.png'))
    plt.show()

    # Create new figure for loss landscape comparison
    plt.figure(figsize=(18, 12))

    for i, lr in enumerate(learning_rates, 1):
        plt.subplot(2, 2, i)
        plt.title(f'Loss Landscape (LR: {lr})')

        # Process VGG_A
        dropout_losses = all_results['VGG_A_Dropout'][lr]['losses_list']
        dropout_min = [np.min(epoch) for epoch in dropout_losses]
        dropout_max = [np.max(epoch) for epoch in dropout_losses]
        plt.fill_between(range(epochs_n), dropout_min, dropout_max,
                         color='blue', alpha=0.2, label='VGG Range')
        plt.plot(dropout_min, 'b-', label='VGG Min')
        plt.plot(dropout_max, 'b--', label='VGG Max')

        # Process VGG_A_BatchNorm
        batchnorm_losses = all_results['VGG_A_BatchNorm'][lr]['losses_list']
        batchnorm_min = [np.min(epoch) for epoch in batchnorm_losses]
        batchnorm_max = [np.max(epoch) for epoch in batchnorm_losses]
        plt.fill_between(range(epochs_n), batchnorm_min, batchnorm_max,
                         color='red', alpha=0.2, label='BatchNorm Range')
        plt.plot(batchnorm_min, 'r-', label='BatchNorm Min')
        plt.plot(batchnorm_max, 'r--', label='BatchNorm Max')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'loss_landscape_comparison.png'))
    plt.show()

# Run experiments and plot results
all_results = run_experiments()
plot_comparison_results(all_results)