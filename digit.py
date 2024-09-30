import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Step 1: Load the MNIST Dataset from CSV
from torchvision.transforms import ToPILImage

class MNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.to_pil = ToPILImage()  # Initialize ToPILImage transformation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image data and label
        image = self.data.iloc[idx, 1:].values.astype('float32')
        label = self.data.iloc[idx, 0]
        
        # Reshape image to 28x28 and normalize
        image = image.reshape(28, 28) / 255.0  # Normalize pixel values to [0, 1]
        
        # Add channel dimension to the image
        image = image[np.newaxis, :, :]  # Shape becomes (1, 28, 28)
        
        # Convert to PyTorch tensor
        image = torch.tensor(image)  # Converts to shape (1, 28, 28)
        
        # Convert to PIL image if transformation is provided
        if self.transform:
            image = self.to_pil(image)  # Convert tensor to PIL image
            image = self.transform(image)  # Apply transformations

        return image.float(), torch.tensor(label)  # Ensure image is of type float



# File path to your CSV dataset
csv_file_path = 'mnist_train.csv'  # Update this to your CSV file path

# Step 2: Create DataLoaders
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
])

dataset = MNISTDataset(csv_file=csv_file_path, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Step 3: Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Convolutional Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling Layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully Connected Layer
        self.fc2 = nn.Linear(128, 10)  # Output Layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply conv1 and activation
        x = self.pool(torch.relu(self.conv2(x)))  # Apply conv2 and activation
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # Apply fc1 and activation
        x = self.fc2(x)  # Apply fc2 (output layer)
        return x

# Step 4: Initialize Model, Loss Function, and Optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training the Model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Step 6: Testing the Model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
