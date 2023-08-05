import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import time
import os.path as osp

# Loading dataset and pre-processing dataset
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),  # MNIST images are 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for single-channel images
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data_mnist', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data_mnist', train=False, download=True, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Change the input channel to 1 channel
resnet50 = models.resnet50(pretrained=False)
num_classes = 10
resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

log_file_path = "/content/gdrive/MyDrive/Project_II/model_logging/resnet_50_mnist.log"

# definition a function to write data to log file
def write_log(log_file_path, content):
    with open(log_file_path, 'a') as log_file:
        log_file.write(content + '\n')

# definition a function to read data from log file
def read_log(log_file_path):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    # Check if log file contains data or being empty
    if len(lines) == 0:
        return 0, None

    # Get the number of epochs has been completed
    num_epochs_completed = len(lines) // 3

    # Checking in the log file if the remaining number of epochs still acceptable
    if len(lines) < 3 or len(lines) % 3 != 0:
        print("Error: File log contains incomplete or invalid information.")
        return num_epochs_completed, None

    try:
        # Get the information of the last epoch
        last_epoch_info = lines[-3:]
        last_epoch_train_loss = float(last_epoch_info[0].split(":")[1].strip())
        last_epoch_train_accuracy = float(last_epoch_info[1].split(":")[1].strip()[:-1])
    except (ValueError, IndexError) as e:
        print("Error: Invalid format or missing information in log file.")
        return num_epochs_completed, None

    return num_epochs_completed, (last_epoch_train_loss, last_epoch_train_accuracy)

# Intialize the array to save training time per each epoch
epoch_times_list = []

# Starting to calculate training model time
start_time = time.time()

# Training model per each epoch
def train(epoch):
    resnet50.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
    write_log(log_file_path, f"Training Loss: {train_loss / len(train_loader):.3f} \t\t Training Accuracy: {100 * correct / total:.2f}%")
    write_log(log_file_path, f"\tElapsed Time: {elapsed_time_str}")

    # Updating total running time of all epochs
    total_elapsed_time = sum([epoch_times[1] for epoch_times in epoch_times_list])
    total_elapsed_time_str = f"{int(total_elapsed_time // 3600)}h {int((total_elapsed_time % 3600) // 60)}m {int(total_elapsed_time % 60)}s"
    write_log(log_file_path, f"\tTotal Elapsed Time: {total_elapsed_time_str}\n{'-'*50}")

    print(f"Epoch {epoch+1}:\nTraining Loss: {train_loss / len(train_loader):.3f}, Accuracy: {100 * correct / total:.2f}%")

def test():
    resnet50.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = resnet50(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    write_log(log_file_path, f"ResNet-50\n{'-'*50}\nEpoch {epoch+1}:\nTest Loss: {avg_loss:.3f} \t\t Accuracy: {accuracy:.2f}%")
    print(f"ResNet-50\n{'-'*50}\nEpoch {epoch+1}:\nTest Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%")

# Moving the model to device
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

# Read data from the last epoch
num_epochs_completed, last_epoch_info = read_log(log_file_path)
start_epoch = num_epochs_completed

for epoch in range(start_epoch, num_epochs):
    test()
    train(epoch)
    scheduler.step()

    #Saving running time result
    end_time = time.time()
    elapsed_time = end_time - start_time
    epoch_times_list.append((epoch + 1, elapsed_time))

    torch.save(resnet50.state_dict(), "/content/gdrive/MyDrive/Project_II/model_logging/resnet50_cifar10_model.pth")