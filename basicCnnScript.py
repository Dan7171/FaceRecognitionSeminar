import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Step 1: Import the necessary libraries and modules

# Step 2: Prepare the data
dataset = CIFAR10(root='path/to/data', train=True, download=True)
data_loader = DataLoader(dataset, batch_size=32, shufflse=True)

# Step 3: Define the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Step 4: Train the model
for epoch in range(10):
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Step 5: Evaluate the model on test data

test_dataset = CIFAR10(root='path/to/data', train=False, download=True)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy on the test set: {correct / total}')