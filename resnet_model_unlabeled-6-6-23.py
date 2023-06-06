import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define the path to the main directory containing the pass, fail, and unlabeled folders
main_directory = '/path/to/main/directory'

# Define the transform to be applied to the input images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the input size expected by ResNet (224x224)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize the image
])

# Define the custom dataset class
class CrystalDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.file_list = []
        
        # Iterate over the pass, fail, and unlabeled directories
        for label in ['pass', 'fail', 'unlabeled']:
            label_directory = os.path.join(directory, label)
            if os.path.exists(label_directory):
                # Iterate over the image files in the pass, fail, or unlabeled directory
                for file_name in os.listdir(label_directory):
                    file_path = os.path.join(label_directory, file_name)
                    self.file_list.append((file_path, label))
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path, label = self.file_list[index]
        image = Image.open(file_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

# Create the ResNet model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # Three output classes: pass, fail, and unlabeled

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model to the device
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the training parameters
batch_size = 32
num_epochs = 10

# Create the training dataset and data loader
train_directory = os.path.join(main_directory, 'train')  # Path to the training crystals directory
train_dataset = CrystalDataset(train_directory, transform=image_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the validation dataset and data loader
val_directory = os.path.join(main_directory, 'val')  # Path to the validation crystals directory
val_dataset = CrystalDataset(val_directory, transform=image_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = torch.tensor([0 if label == 'pass' else 1 if label == 'fail' else 2 for label in labels], dtype=torch.long).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = torch.tensor([0 if label == 'pass' else 1 if label == 'fail' else 2 for label in labels], dtype=torch.long).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    # Print epoch results
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - "
          f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'resnet_model.pth')
