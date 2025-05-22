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

from models.vgg import VGG_A,VGG_A_Dropout
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader
matplotlib.use('TkAgg')

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    img = X[0] / 2 + 0.5  # 反标准化（如果使用 Normalize(mean=0.5, std=0.5)）
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # (C, H, W) -> (H, W, C)
    plt.title(f"Label: {y[0].item()}")  # 显示标签
    plt.savefig("sample_image.png")
    break
    ## --------------------



# This function is used to calculate the accuracy of model classification
def get_accuracy(model,loader):
    ## --------------------
    # Add code as needed
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
    ## --------------------
    return correct/total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss_list.append(loss.item())
            ## --------------------


            loss.backward()
            grad=model.classifier[4].weight.grad.clone()
            optimizer.step()
        grads.append(grad.cpu())
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader)
        if val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy_epoch = epoch
            max_val_accuracy = val_accuracy_curve[epoch]
        losses_list.append(loss_list)
        learning_curve[epoch] = np.mean(loss_list)

        print(f'Epoch {epoch + 1}/{epochs_n}:')
        print(f'Train Accuracy: {train_accuracy_curve[epoch]:.2f}%')
        print(f'Val Accuracy: {val_accuracy_curve[epoch]:.2f}%')
        print(f'Training Loss: {learning_curve[epoch]:.4f}')

    # Test your model and save figure here (not required)
    # remember to use model.eval()
    ## --------------------
    # Add code as needed
    model.eval()
    display.clear_output(wait=True)
    f, axes = plt.subplots(1, 2, figsize=(15, 3))

    learning_curve[epoch] /= batches_n
    axes[0].plot(learning_curve)
    display.clear_output(wait=True)

    # Plot 1: Training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.plot(train_accuracy_curve, label='Train Accuracy')
    plt.plot(val_accuracy_curve, label='Validation Accuracy')
    plt.scatter(max_val_accuracy_epoch, max_val_accuracy, c='r',
                label=f'Max Val Acc: {max_val_accuracy:.2f}')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # 固定准确率Y轴范围
    plt.legend()
    plt.grid(True)

    # Plot 2: Training loss
    plt.subplot(1, 2, 2)
    plt.cla()
    plt.plot(learning_curve, label='Training Loss')
    plt.title('Training Loss Curve(VGG_A_Dropout)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 5)  # 固定Loss Y轴范围
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'accuracy.png'))
    plt.show()

    ## --------------------

    return losses_list, grads


# Train your model
# feel free to modify
epo = 20
loss_save_path = ''
grad_save_path = ''

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
loss_array = np.array(loss)
grads_array = np.array(grads)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss_array, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads_array.reshape(-1,grads_array.size//grads_array.shape[0]), fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
for epoch in range(epo):
    min_curve.append(np.min(loss[epoch]))
    max_curve.append(np.max(loss[epoch]))
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve,max_curve):
    ## --------------------
    # Add your code
    plt.figure(figsize=(10, 6))
    plt.ylim(0,3)

    # Create x-axis values (epochs)
    epochs = range(1, len(min_curve) + 1)

    # Plot the area between min and max curves
    plt.fill_between(epochs, min_curve, max_curve,
                     color='skyblue', alpha=0.4,
                     label='Loss Range')

    # Plot the min and max curves
    plt.plot(epochs, min_curve, 'b-', linewidth=2, label='Min Loss')
    plt.plot(epochs, max_curve, 'r-', linewidth=2, label='Max Loss')

    # Add markers for important points
    min_idx = np.argmin(min_curve)
    max_idx = np.argmax(max_curve)
    plt.scatter(min_idx + 1, min_curve[min_idx],
                color='blue', s=100, zorder=5,
                label=f'Global Min: {min_curve[min_idx]:.2f}')
    plt.scatter(max_idx + 1, max_curve[max_idx],
                color='red', s=100, zorder=5,
                label=f'Global Max: {max_curve[max_idx]:.2f}')

    # Customize the plot
    plt.title('Training Loss Landscape(VGG_A)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'loss_landscape_tmp.png'))
    plt.show()
    ## --------------------
plot_loss_landscape(min_curve,max_curve)