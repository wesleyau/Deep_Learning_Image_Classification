import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import numpy as np

main_directory = '/data/wesley/data2/dataset_tx_pseudo'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This transformation has been updated to handle grayscale images.
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
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
        image_paths.sort()

        # Load the grayscale images using np.load
        images = [np.load(p, allow_pickle=True) for p in image_paths]

        # Convert images to tensors and stack along a new dimension
        image_tensor = torch.stack([self.transform(img) for img in images])

        if label == 'pass':
            label = 0
        elif label == 'fail':
            label = 1
        else:  # unlabeled
            label = 2  # use a specific integer for 'unlabeled'
                
        return image_tensor, label

class MultiStreamResNet(nn.Module):
    def __init__(self, num_streams=4):
        super(MultiStreamResNet, self).__init__()
        self.streams = nn.ModuleList([resnet50() for _ in range(num_streams)])

        # Change the first convolution layer and last layer in each stream
        for stream in self.streams:
            stream.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer here

# Define the training parameters
batch_size = 50
num_epochs = 10

# Create the training dataset and data loader
train_directory = os.path.join(main_directory, 'train')
train_dataset = CrystalDataset(train_directory, transform=image_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the validation dataset and data loader
val_directory = os.path.join(main_directory, 'val')
val_dataset = CrystalDataset(val_directory, transform=image_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Lists to store training history
train_loss_history = []  # Change to empty tensors
train_acc_history = []
val_loss_history = []  # Change to empty tensors
val_acc_history = []

# Choose the number of batches for which to accumulate gradients
accumulation_steps = 4

# Change criterion to a placeholder, we'll initialize it properly inside the training loop
criterion = None

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    # Count samples in each class for each epoch
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, labels in train_loader:
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        class_counts.update(dict(zip(unique, counts)))
    
    # Count unlabeled samples as half for each class
    # The reason for initially counting unlabeled crystals as half pass and half fail is based on the assumption that, 
    # before we have any other information, unlabeled data is equally likely to be either 'pass' or 'fail'.
    # As the training progresses, the pseudo-labeling process will assign a class to the unlabeled data, 
    # and the class counts (and consequently class weights) will be updated based on these new labels.
    unlabeled_count = class_counts.pop(2)
    class_counts[0] += unlabeled_count // 2
    class_counts[1] += unlabeled_count // 2
    
    # Calculate class weights and create the criterion
    total_count = sum(class_counts.values())
    class_weights = [total_count / class_counts[i] for i in range(2)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        # Convert the labels to a tensor if they are not
        if type(labels) is list:
            labels = torch.Tensor(labels)

        # Create masks for labeled and unlabeled data
        mask_labeled = (labels != 2)
        mask_unlabeled = (labels == 2)

        images_labeled = images[mask_labeled].to(device)
        labels_labeled = labels[mask_labeled].long().to(device)

        # Train on the labeled data
        optimizer.zero_grad()
        outputs = model(images_labeled)
        loss = criterion(outputs, labels_labeled)
        loss.backward()

        # Only perform an optimization step every `accumulation_steps` batches
        if (i + 1) % accumulation_steps == 0 or i + 1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels_labeled.size(0)
        correct += (predicted == labels_labeled).sum().item()

        running_loss += loss.item() * images_labeled.size(0)

        # Pseudo-labeling for unlabeled data
        confidence_threshold = 0.80
        if mask_unlabeled.sum() > 0:
            images_unlabeled = images[mask_unlabeled].to(device)
            outputs_unlabeled = model(images_unlabeled)
            probabilities_unlabeled = torch.nn.functional.softmax(outputs_unlabeled, dim=1)
            max_probs, pseudo_labels = torch.max(probabilities_unlabeled.data, 1)
            confident_indices = max_probs > confidence_threshold
            if confident_indices.sum() > 0:
                loss_unlabeled = criterion(outputs_unlabeled[confident_indices], pseudo_labels[confident_indices])
                loss_unlabeled.backward()
                optimizer.step()
                running_loss += loss_unlabeled.item() * confident_indices.sum()
                #print("Psuedo-labeling used")
            #else:
                #print("Pseudo-labeling skipped")
                
        # Update class_counts and class_weights after pseudo-labeling
        if mask_unlabeled.sum() > 0:
            unique, counts = torch.unique(pseudo_labels[confident_indices], return_counts=True)
            for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                class_counts[u] += c

            total_count = sum(class_counts.values())
            class_weights = [total_count / class_counts[i] for i in range(2)]
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_history.append(torch.tensor(train_loss))  # Append as tensors
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

            # Calculate accuracy
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()
            running_val_loss += loss_val.item() * images_val.size(0)

    val_loss = running_val_loss / total_val
    val_acc = correct_val / total_val
    val_loss_history.append(torch.tensor(val_loss))  # Append as tensors
    val_acc_history.append(val_acc)
    print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")

# Convert the history lists to tensors
train_loss_history = torch.stack(train_loss_history)
val_loss_history = torch.stack(val_loss_history)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history.cpu().numpy(), label='Training Loss')
plt.plot(val_loss_history.cpu().numpy(), label='Validation Loss')
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
