# need to change the NaNs for the background to 0 or 255 (black or white) 
# before loading them into the model

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define the ResNet model
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Load the pre-trained ResNet-50 model
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Replace the last fully connected layer

    def forward(self, x):
        return self.resnet(x)

# Set the number of classes
num_classes = 3

# Create an instance of the ResNet model
model = ResNetModel(num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Training steps
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation steps
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch: {epoch+1}, Accuracy: {accuracy:.2f}%")
