import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import time
import os
import os.path as osp

# Loading dataset and pre-processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

trainset = CIFAR100(root='./data_cifar100', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR100(root='./data_cifar100', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Buidling VGG16 network
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = VGG16()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

num_epochs = 100

log_file_path = "/content/gdrive/MyDrive/Project_II/model_logging/VGG_16_cifar100.log"

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

    # Check if the remained lines of the log file
    if len(lines) < 3 or len(lines) % 3 != 0:
        print("Error: File log contains incomplete or invalid information.")
        return num_epochs_completed, None

    try:
        # Get the information of last epoch
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

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    net.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    write_log(log_file_path, f"VGG-16\n{'-'*50}\nEpoch {epoch + 1}:\n\tTrain loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(trainloader)}, Train Accuracy: {train_accuracy:.2f}%")

    # Evaluate the model in testing dataset and calculate the accuracy
    net.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_accuracy = 100 * correct_test / total_test
    write_log(log_file_path, f"VGG-16\n{'-'*50}\nEpoch {epoch + 1}:\n\tAccuracy on test set: {test_accuracy:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - epoch_start_time
    elapsed_time_str = f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s"
    epoch_times_list.append((epoch + 1, elapsed_time))

    # Calculate total elapsed time
    total_elapsed_time = sum([epoch_times[1] for epoch_times in epoch_times_list])
    total_elapsed_time_str = f"{int(total_elapsed_time // 3600)}h {int((total_elapsed_time % 3600) // 60)}m {int(total_elapsed_time % 60)}s"
    write_log(log_file_path, f"\tElapsed Time: {elapsed_time_str}")
    write_log(log_file_path, f"\tTotal Elapsed Time: {total_elapsed_time_str}\n{'-'*50}")
    print(f"Accuracy on test set: {test_accuracy:.2f}%")