import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torchvision.transforms as transforms

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Loading dataset
train_dataset = MNIST(root='./data_mnist', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data_mnist', train=False, download=True, transform=transform)

# Definition train_loader và test_loader
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Teacher Model
class ConvNeXt(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNeXt, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(64 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Student Model
class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define distillation loss function
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs_student, outputs_teacher):
        # Apply temperature scaling to student model's outputs
        outputs_student = outputs_student / self.temperature
        # Compute distillation loss
        loss = self.loss_fn(torch.log_softmax(outputs_student, dim=1), torch.softmax(outputs_teacher, dim=1))
        return loss

# Initialize teacher and student models
teacher_model = ConvNeXt()
student_model = MobileNet()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the device
teacher_model.to(device)
student_model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
distillation_loss_fn = DistillationLoss(temperature=3.0)

# Training loop
num_epochs = 20
batch_size = 10

for epoch in range(num_epochs):
    student_model.train()
    teacher_model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data to the device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        # Forward pass through student model
        student_outputs = student_model(images)

        # Compute distillation loss
        loss = distillation_loss_fn(student_outputs, teacher_outputs)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    # Compute average loss for the epoch
    epoch_loss = total_loss / total_samples

    # Evaluate on test dataset
    student_model.eval()
    total_mae = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass through student model
            student_outputs = student_model(images)

            # Compute MAE
            mae = torch.mean(torch.abs(student_outputs - labels))
            total_mae += mae.item() * images.size(0)
            total_test_samples += images.size(0)

    # Compute average MAE for the test dataset
    test_mae = total_mae / total_test_samples

    # Print the average loss and test MAE for the epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Test MAE: {test_mae:.4f}")