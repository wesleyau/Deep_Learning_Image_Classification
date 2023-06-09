import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the main directory and device
main_directory = '/data/wesley/data2/dataset_tx'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transform to be applied to the input images
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # these are just the ImageNet mean and SD values. 
    # I will need to calculate the eman and SD of the pixel values across all images in my dataset and change this
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class CrystalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = ['pass', 'fail', 'unlabeled']
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
        # sort the image paths so that the channels are always in the same order
        image_paths.sort()

        # Load the 4 images and concatenate them to create a 4-channel image
        images = [Image.open(p) for p in image_paths]
        image_tensor = torch.stack([self.transform(img) for img in images])

        # Merge the 4 images into a single 4-channel image
        image_tensor = image_tensor.view(4, 256, 256)

        if label == 'pass':
            label = 0
        elif label == 'fail':
            label = 1
        else:  # unlabeled
            label = 'unlabeled'
        return image_tensor, label
    
# Create the ResNet model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # changed here
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the training parameters
batch_size = 32
num_epochs = 10

# Create the training dataset and data loader
train_directory = os.path.join(main_directory, 'train')
train_dataset = CrystalDataset(train_directory, transform=image_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the validation dataset and data loader
val_directory = os.path.join(main_directory, 'val')
val_dataset = CrystalDataset(val_directory, transform=image_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lists to hold the losses and accuracies for each epoch
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop over the training data
    for images, labels in train_loader:
        # Select only the labeled data
        mask_labeled = (labels != 'unlabeled')
        images_labeled = images[mask_labeled]
        labels_labeled = torch.tensor([0 if label == 'pass' else 1 for label in labels[mask_labeled]], dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(images_labeled)
        loss = criterion(outputs, labels_labeled)
        loss.backward()
        optimizer.step()
        

        # calculate statistics only for labeled data
        _, predicted = torch.max(outputs_labeled.data, 1)
        total += labels_labeled.size(0)
        correct += (predicted == labels_labeled).sum().item()

        running_loss += loss.item() * images_labeled.size(0)

        # Pseudo-labeling for unlabeled data
        # This contains a confidence threshold to decide when to use pseudo-labels
        # its implemented by taking the maximum probability from the softmax ouput and checking if it's above a certain threshold
        # if it is, the pseudo-label is used for training, otherwise, its ignored
        mask_unlabeled = (labels == 'unlabeled')
        confidence_threshold = 0.95  # or any other value you find appropriate of how confident you want ot be in pseudolabeling
        if mask_unlabeled.sum() > 0:
            images_unlabeled = images[mask_unlabeled].to(device)
            outputs_unlabeled = model(images_unlabeled)
            probabilities_unlabeled = torch.nn.functional.softmax(outputs_unlabeled, dim=1)
            max_probs, pseudo_labels = torch.max(probabilities_unlabeled.data, 1)
            confident_indices = max_probs > confidence_threshold
            if confident_indices.sum() > 0:
                print("Pseudo-label used")
                loss_unlabeled = criterion(outputs_unlabeled[confident_indices], pseudo_labels[confident_indices])
                loss_unlabeled.backward()
            else:
                print("Pseudo-label skipped")

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # Loop over the validation data
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - "
          f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

# Save model
torch.save(model.state_dict(), 'resnet_model.pth')

# Plot loss and accuracy for training and validation
epochs = np.arange(num_epochs) + 1

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
