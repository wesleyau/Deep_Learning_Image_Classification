import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

main_directory = '/data/wesley/data2/dataset_tx'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This transformation has been updated to handle 3 color channels instead of 1.
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CrystalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = ['pass', 'fail']
        for label in classes:
            class_dir = os.path.join(root_dir, label)
            if os.path.exists(class_dir):
                for crystal in os.listdir(class_dir):
                    crystal_dir = os.path.join(class_dir, crystal)
                    image_paths = [os.path.join(crystal_dir, img) for img in os.listdir(crystal_dir)]
                    self.samples.append((image_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, label = self.samples[idx]
        image_paths.sort()

        # Load the 4 images and process them individually
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # Convert images to tensors and stack along a new dimension
        image_tensor = torch.stack([self.transform(img) for img in images])

        if label == 'pass':
            label = 0
        elif label == 'fail':
            label = 1
                
        return image_tensor, label

class MultiStreamResNet(nn.Module):
    def __init__(self, num_streams=4):
        super(MultiStreamResNet, self).__init__()
        self.streams = nn.ModuleList([resnet50() for _ in range(num_streams)])

        # Change the first convolution layer and last layer in each stream
        for stream in self.streams:
            stream.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = stream.fc.in_features
            stream.fc = nn.Linear(num_ftrs, 2) 
            
    def forward(self, x):
        # x shape: (batch_size, num_streams, channels, height, width)
        outputs = [stream(x[:, i]) for i, stream in enumerate(self.streams)]
            
        # Average the outputs
        outputs = torch.stack(outputs, dim=2)  # outputs shape: (batch_size, num_classes, num_streams)
        outputs = torch.mean(outputs, dim=2)  # outputs shape: (batch_size, num_classes)
            
        return outputs

# Create the model
model = MultiStreamResNet(num_streams=4)

# Check if multiple GPUs are available and wrap the model using DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

# Now move the model to the device
model = model.to(device)

# Define the training parameters
batch_size = 50
num_epochs = 30

# Create the training dataset and data loader
train_directory = os.path.join(main_directory, 'train')
train_dataset = CrystalDataset(train_directory, transform=image_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Calculate the number of samples in each class
num_pass = len([sample for sample in train_dataset if sample[1] == 0])
num_fail = len([sample for sample in train_dataset if sample[1] == 1])
total_samples = num_pass + num_fail

# Calculate class weights
class_weights = torch.FloatTensor([total_samples / num_pass, total_samples / num_fail]).to(device)

# Update the criterion with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create the validation dataset and data loader
val_directory = os.path.join(main_directory, 'val')
val_dataset = CrystalDataset(val_directory, transform=image_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lists to store training history
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f} - Training Accuracy: {train_acc:.4f}")

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)

            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()
            running_val_loss += loss_val.item() * images_val.size(0)

    val_loss = running_val_loss / total_val
    val_acc = correct_val / total_val
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Losses over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
