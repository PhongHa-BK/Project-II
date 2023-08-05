import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import time
import os.path as osp

# Definition of the GhostModule class
class GhostModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, ratio=2):
        super(GhostModule, self).__init__()
        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels // ratio, kernel_size, bias=False),
            nn.BatchNorm2d(output_channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(output_channels // ratio, output_channels, kernel_size, groups=output_channels // ratio, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # The first GhostModule block
        x2 = self.cheap_operation(x1)  # The second GhostModule block
        out = torch.cat([x1, x2], dim=1)
        return out[:, :x.shape[1], :, :]

# Create SE layer, which associate with GhostBottleneck
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Create class GhostBottleneck
class GhostBottleneck(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size, stride, use_se=True):
        super(GhostBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.ghost = GhostModule(hidden_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.use_se = use_se

        if self.use_se:
            self.se = SELayer(output_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.ghost(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.stride == 1 and identity.shape[1] == out.shape[1]:
            identity = nn.functional.interpolate(identity, size=out.shape[2:], mode='nearest')
            out += identity

        return out

# Callback GhostBottleneck and GhostModule in class GhostNet
class GhostNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),  # Change the input channel to 1 (grayscale)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(16, 16, 1, stride=1)
        self.stage2 = self._make_stage(16, 24, 2, stride=2)
        self.stage3 = self._make_stage(24, 40, 3, stride=2, use_se=True)  # Using SE layer 
        self.stage4 = self._make_stage(40, 80, 3, stride=2, use_se=True)
        self.stage5 = self._make_stage(80, 96, 2, stride=1, use_se=True)
        self.stage6 = self._make_stage(96, 192, 4, stride=2, use_se=True)
        self.stage7 = self._make_stage(192, 320, 1, stride=1, use_se=True)

        self.conv9 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def _make_stage(self, input_channels, output_channels, num_blocks, stride, use_se=False):  # Thêm tham số use_se
        layers = []
        layers.append(GhostBottleneck(input_channels, input_channels // 2, output_channels, 3, stride, use_se))  # Sử dụng SE trong GhostBottleneck
        for _ in range(1, num_blocks):
            layers.append(GhostBottleneck(output_channels, output_channels // 2, output_channels, 3, 1, use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.conv9(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Adjust the number of classes to 10 for MNIST
ghostnet = GhostNet(num_classes=10)

# Write data into the log file
def write_log(log_file_path, content):
    with open(log_file_path, 'a') as log_file:
        log_file.write(content + '\n')

# Create read_log function to read data from the log file:
def read_log(log_file_path):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    # Check if file log contains data
    if len(lines) == 0:
        return 0, None

    # Get the number of epochs has been completed
    num_epochs_completed = len(lines) // 4

    # Checking in the log file if the remaining number of epochs still acceptable 
    if len(lines) < 3 or len(lines) % 3 != 0:
        print("Error: File log contains incomplete or invalid information.")
        return num_epochs_completed, None

    try:
        # Get the information of the last epoch
        last_epoch_info = lines[-5:]
        last_epoch_train_loss = float(last_epoch_info[0].split(":")[1].strip())
        last_epoch_train_accuracy = float(last_epoch_info[1].split(":")[1].strip()[:-1])
    except (ValueError, IndexError) as e:
        print("Error: Invalid format or missing information in log file.")
        return num_epochs_completed, None

    return num_epochs_completed, (last_epoch_train_loss, last_epoch_train_accuracy)

# Load MNIST dataset and pre-processing data
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),  # MNIST images are 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for single-channel images
])

test_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data_mnist', train=True, download=True, transform=train_transform)
test_dataset = MNIST(root='./data_mnist', train=False, download=True, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Definition loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ghostnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Intialize the array to save training time per each epoch
epoch_times_list = []

# Starting to calculate training model time
start_time = time.time()

def train_model(num_epochs, ghostnet, train_loader, test_loader, criterion, optimizer, device, log_file_path):
    # Read the data from the log file
    num_epochs_completed, last_epoch_info = read_log(log_file_path)

    # start_epoch is recorded from the last training turn, if there is the first training turn so num_epochs_completed = 0
    start_epoch = num_epochs_completed + 1

    # Adding information of last epoch
    if last_epoch_info:
        last_epoch_train_loss, last_epoch_train_accuracy, last_epoch_test_loss, last_epoch_test_accuracy = last_epoch_info
        write_log(log_file_path, f"GhostNet V2\n{'-'*50}\nEpoch {num_epochs_completed + 1}:\n\tTrain loss: {last_epoch_train_loss:.3f}, Train Accuracy: {last_epoch_train_accuracy:.2f}%")
        write_log(log_file_path, f"\tTest Loss: {last_epoch_test_loss:.3f}, Test Accuracy: {last_epoch_test_accuracy:.2f}%")
        write_log(log_file_path, f"\tElapsed Time: {elapsed_time_str}")

    # Initialize total elapsed time
    total_elapsed_time = 0

    # Starting time for training process
    start_time = time.time()

    # Training model from the last epoch.
    for epoch in range(start_epoch, num_epochs):
        # Start time for current epoch
        epoch_start_time = time.time()

        # Training model for training dataset
        ghostnet.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward passing
            outputs = ghostnet(images)
            loss = criterion(outputs, labels)

            # Backward passing and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updating the total loss and number of correct predictions
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        # Evaluate mode for testing dataset
        ghostnet.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = ghostnet(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()

        # Calculating the accuracy 
        train_loss = train_loss / len(train_dataset)
        train_accuracy = train_correct / len(train_dataset) * 100
        test_loss = test_loss / len(test_dataset)
        test_accuracy = test_correct / len(test_dataset) * 100

        # Saving the training model
        torch.save(ghostnet.state_dict(), "GhostNet_mnist_model.pth")

        # Saving output results in to log file
        write_log(log_file_path, f"GhostNet V2\n{'-'*50}\nEpoch {epoch+1}:\n\tTraining Loss: {train_loss:.3f}, Testing Accuracy: {train_accuracy:.2f}%")
        write_log(log_file_path, f"\tTest Loss: {train_loss:.3f}, Test Accuracy: {train_accuracy:.2f}%")
        
        # Estimate the training time of each epoch
        end_time = time.time()
        elapsed_time = end_time - epoch_start_time
        elapsed_time_str = f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s"
        epoch_times_list.append((epoch + 1, elapsed_time))

        # Calculate total training time
        total_elapsed_time = sum([epoch_times[1] for epoch_times in epoch_times_list])
        total_elapsed_time_str = f"{int(total_elapsed_time // 3600)}h {int((total_elapsed_time % 3600) // 60)}m {int(total_elapsed_time % 60)}s"
        write_log(log_file_path, f"\tElapsed Time: {elapsed_time_str}")
        write_log(log_file_path, f"\tTotal Elapsed Time: {total_elapsed_time_str}\n{'-'*50}")

        print(f"GhostNet V2\n{'-'*50}\nEpoch {epoch+1}:\n\tLoss: {train_loss:.3f}, Accuracy: {train_accuracy:.2f}%")

# Load GhostNet model
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ghostnet.to(device)

log_file_path = "/content/gdrive/MyDrive/Project_II/model_logging/ghost_net_V2_mnist.log"

# Read the information from the log file and continue the training loop from the neareast completed epoch.
num_epochs_completed, last_epoch_info = read_log(log_file_path)
start_epoch = num_epochs_completed

# Implement training model
train_model(num_epochs, ghostnet, train_loader, test_loader, criterion, optimizer, device, log_file_path)