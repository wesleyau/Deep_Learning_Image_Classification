import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define your custom dataset for unlabeled data
class UnlabeledDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.file_list = os.listdir(data_folder)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # Load and preprocess the unlabeled image
        image_path = os.path.join(self.data_folder, self.file_list[index])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        
        return image

# Set the path to your unlabeled data folder
unlabeled_data_folder = '/path/to/unlabeled/data'

# Define data transformations for augmentation
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
])

# Create an instance of your custom dataset with unlabeled data
unlabeled_dataset = UnlabeledDataset(unlabeled_data_folder, transform=data_transform)

# Create a data loader for the unlabeled dataset
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=4)

# Load the pre-trained ResNet model
resnet = torchvision.models.resnet50(pretrained=True)
num_features = resnet.fc.in_features

# Modify the last fully connected layer for your classification task
resnet.fc = nn.Linear(num_features, num_classes)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)
resnet.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for images in unlabeled_dataloader:
        images = images.to(device)
        
        # Forward pass
        outputs = resnet(images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the average loss for the epoch
    epoch_loss = running_loss / len(unlabeled_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# Fine-tuning on labeled data
labeled_data_folder = '/path/to/labeled/data'

# Create a labeled dataset and dataloader
labeled_dataset = torchvision.datasets.ImageFolder(labeled_data_folder, transform=data_transform)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=4)

# Reset the optimizer and train only the last layer
optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

# Training loop for fine-tuning
resnet.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for images, labels in labeled_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = resnet(images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the average loss for the epoch
    epoch_loss = running_loss / len(labeled_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(resnet.state_dict(), "resnet_model.pth")
