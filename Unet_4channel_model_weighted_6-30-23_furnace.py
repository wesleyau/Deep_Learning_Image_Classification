import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch.nn as nn
import torch.nn.functional as F

class CrystalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        
        # Look for specific image types in the directory
        image_types = ['Am_image', 'Co_image', 'Elin_image', 'Eres_image']
        images = []
        for image_type in image_types:
            for file in os.listdir(sample_dir):
                if image_type in file:
                    image_path = os.path.join(sample_dir, file)
                    image = Image.open(image_path).convert('L')  # Ensure the image is grayscale
                    if self.transform:
                        image = self.transform(image).squeeze()
                    images.append(image)  # Add image without adding a channel dimension

        images = torch.stack(images, dim=0)  # Stack along the new channel dimension
        
        filename = self.samples[idx]
        furnace_number = int(filename[0])
        label = 1 if "Pass" in filename else 0
        
        return {"images": images, "furnace_number": furnace_number}, label


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[46.25611771855562], std=[0.485897429579875]),  # replace 'your_mean' and 'your_std' with your dataset's mean and std
])

train_dataset = CrystalDataset(root_dir="/data/wesley/data2_6-27-23/train_test/TY/train", transform=transform)
test_dataset = CrystalDataset(root_dir="/data/wesley/data2_6-27-23/train_test/TY/test", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if not bilinear, use the normal transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CrystalClassifier(nn.Module):
    def __init__(self):
        super(CrystalClassifier, self).__init__()
        self.inc = double_conv(4, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(512, 512)
        )
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = nn.Linear(64, 2)
        self.fc1 = nn.Linear(3, 2)

    def forward(self, x, furnace_number):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.mean(x, dim=[2,3])  # global average pooling (1D)

        # expand dimensions of furnace_number to concatenate it with x
        furnace_number = furnace_number.view(-1, 1).float()  # reshape and convert to float
        x = torch.cat((x, furnace_number), dim=1)  # concatenate along the channel dimension
        
        x = self.fc1(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CrystalClassifier().to(device)

# Check if multiple GPUs are available and wrap the model using DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)

# Calculate weights
num_total = len(train_dataset)  
num_pass = sum([1 for _, label in train_dataset if label == 1])
num_fail = num_total - num_pass
weights = torch.tensor([num_total / num_fail, num_total / num_pass], dtype=torch.float).to(device)

# Pass the weights to the loss function
criterion = torch.nn.CrossEntropyLoss(weight=weights)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Initialize lists to store loss and accuracy values for plotting
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []

# Train the model
for epoch in range(15):
    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()
    for i, (data, labels) in enumerate(train_dataloader, 0):
        inputs, furnace_number = data["images"].to(device), data["furnace_number"].to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, furnace_number)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_running_loss / len(train_dataloader)
    train_acc = 100 * train_correct / train_total

    # Start validation
    model.eval()
    valid_running_loss = 0.0
    valid_correct = 0
    valid_total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_dataloader, 0):
            inputs, furnace_number = data["images"].to(device), data["furnace_number"].to(device)
            labels = labels.to(device)

            outputs = model(inputs, furnace_number)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            valid_running_loss += loss.item()
            valid_correct += (predicted == labels).sum().item()
            valid_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    valid_loss = valid_running_loss / len(test_dataloader)
    valid_acc = 100 * valid_correct / valid_total

    # Print loss and accuracy for this epoch
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc}")

    # Append loss and accuracy to lists for plotting
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

# After training, plot loss and accuracy for training and validation sets
plt.figure()
plt.plot(train_loss_list, label='TX Training Loss')
plt.plot(valid_loss_list, label='TX Validation Loss')
plt.title('Loss Plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(train_acc_list, label='TX Training Accuracy')
plt.plot(valid_acc_list, label='TX Validation Accuracy')
plt.title('Accuracy Plot')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Display confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
