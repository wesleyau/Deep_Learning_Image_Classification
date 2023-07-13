import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch import nn, optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

# Generate a timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Image transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.furnace_nums = []
        
        classes = os.listdir(root_dir)
        for i, class_ in enumerate(classes):
            class_dir = os.path.join(root_dir, class_)
            furnaces = os.listdir(class_dir)
            for furnace in furnaces:
                furnace_num = int(furnace.split("_")[0][0])  # extract first character as furnace number
                furnace_dir = os.path.join(class_dir, furnace)
                imgs = []
                for img_name in sorted(os.listdir(furnace_dir)):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(furnace_dir, img_name)
                        imgs.append(img_path)
                # Group every two images together
                for j in range(0, len(imgs), 2):
                    self.imgs.append([imgs[j], imgs[j + 1]])
                    self.labels.append(i)  # assign label based on class
                    self.furnace_nums.append(furnace_num)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img1 = self.transform(Image.open(self.imgs[index][0]))
        img2 = self.transform(Image.open(self.imgs[index][1]))
        img = torch.cat([img1, img2], dim=0)  # concatenate along the channel dimension
        label = self.labels[index]
        furnace_num = torch.tensor([self.furnace_nums[index]], dtype=torch.float32)  # pass as tensor
        return img, furnace_num, label

# Load the data
train_data = CustomImageFolder('/data/wesley/2_data/train_test/TY/train', transform=transform)
val_data = CustomImageFolder('/data/wesley/2_data/train_test/TY/val', transform=transform)

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Define the model
model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)

# Change the first layer to accept the two-channel images
model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Change the last layer to accept the furnace number along with the CNN features
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # adjust hidden layer size as required
    nn.ReLU(),
    nn.Linear(512, 2)
)

class CustomModel(nn.Module):
    def __init__(self, base_model, use_furnace_num=True):
        super().__init__()
        self.base_model = base_model
        self.use_furnace_num = use_furnace_num
        if use_furnace_num:
            self.fc = nn.Linear(3, 2)
        else:
            self.fc = nn.Linear(2, 2)

    def forward(self, x, furnace_num):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        if self.use_furnace_num:
            x = torch.cat([x, furnace_num], dim=1)
        x = self.fc(x)
        return x

model_with_furnace_num = CustomModel(model, use_furnace_num=True)
model_without_furnace_num = CustomModel(model, use_furnace_num=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the loss function and the optimizer
weights = [0.83, 0.17]  # class 0 is "Passes" and class 1 is "Fails"
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define the optimizer
optimizer_with = optim.Adagrad(model_with_furnace_num.parameters(), lr=0.001)
optimizer_without = optim.Adagrad(model_without_furnace_num.parameters(), lr=0.001)

# Define the path to the output directory
output_dir = f'/data/wesley/2_data/model_outputs/TY/Adagrad_0.7_0.3_{timestamp}'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Define the learning rate scheduler
scheduler_with = optim.lr_scheduler.ReduceLROnPlateau(optimizer_with, 'min', patience=3, factor=0.1)
scheduler_without = optim.lr_scheduler.ReduceLROnPlateau(optimizer_without, 'min', patience=3, factor=0.1)

models_dict = {'with_furnace': {'model': model_with_furnace_num, 
                                'optimizer': optimizer_with, 
                                'scheduler': scheduler_with, 
                                'roc': [], 
                                'best_model_wts': None},
               'without_furnace': {'model': model_without_furnace_num, 
                                   'optimizer': optimizer_without, 
                                   'scheduler': scheduler_without, 
                                   'roc': [], 
                                   'best_model_wts': None}}

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    for key in models_dict.keys():
        models_dict[key]['model'] = nn.DataParallel(models_dict[key]['model'])

for key in models_dict.keys():
    models_dict[key]['model'] = models_dict[key]['model'].to(device)

train_acc = []
train_loss = []
val_acc = []
val_loss = []

# Training loop
for epoch in range(10):  # 10 epochs, adjust as needed
    epoch_train_loss = []
    epoch_train_corrects = 0
    epoch_val_loss = []
    epoch_val_corrects = 0
    
    for key in models_dict.keys():
        model = models_dict[key]['model']
        optimizer = models_dict[key]['optimizer']
        scheduler = models_dict[key]['scheduler']

        model.train()
        for inputs, furnace_nums, labels in train_loader:
            inputs, furnace_nums, labels = inputs.to(device), furnace_nums.float().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, furnace_nums)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            epoch_train_loss.append(loss.item())
            epoch_train_corrects += torch.sum(preds == labels.data)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, furnace_nums, labels in val_loader:
                inputs, furnace_nums, labels = inputs.to(device), furnace_nums.float().to(device), labels.to(device)
                outputs = model(inputs, furnace_nums)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loss = criterion(outputs, labels)
                
                # Track validation loss and accuracy
                epoch_val_loss.append(loss.item())
                epoch_val_corrects += torch.sum(preds == labels.data)
                
        # Calculate average losses
        avg_train_loss = np.mean(epoch_train_loss)
        avg_val_loss = np.mean(epoch_val_loss)

        # Calculate accuracies
        train_accuracy = epoch_train_corrects.double() / len(train_loader.dataset)
        val_accuracy = epoch_val_corrects.double() / len(val_loader.dataset)

        print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy))

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        models_dict[key]['roc'].append(roc_auc)

        # Calculate average validation loss for the current epoch
        avg_val_loss = np.mean(epoch_val_loss)

        # Adjust the learning rate based on the average validation loss
        scheduler.step(avg_val_loss)

        # Add averages to accuracy and loss lists
        train_acc.append(epoch_train_corrects / len(train_loader))
        train_loss.append(np.mean(epoch_train_loss))
        val_acc.append(epoch_val_corrects / len(val_loader))
        val_loss.append(avg_val_loss)


# Saving the trained model
for key in models_dict.keys():
    torch.save(models_dict[key]['model'].state_dict(), os.path.join(output_dir, f'model_{key}.pt'))

# Generate ROC curve for the model with furnace numbers
# Calculate the probabilities of the predictions
y_score_with_furnace = model_with_furnace_num(inputs, furnace_nums)
y_prob_with_furnace = torch.nn.functional.softmax(y_score_with_furnace, dim=1)
y_prob_with_furnace = y_prob_with_furnace.detach().cpu().numpy()

# Calculate the ROC curve and AUC
fpr_with_furnace, tpr_with_furnace, _ = roc_curve(labels.cpu().numpy(), y_prob_with_furnace[:, 1])
roc_auc_with_furnace = auc(fpr_with_furnace, tpr_with_furnace)

# Generate ROC curve for the model without furnace numbers
# Calculate the probabilities of the predictions
y_score_without_furnace = model_without_furnace_num(inputs, furnace_nums)
y_prob_without_furnace = torch.nn.functional.softmax(y_score_without_furnace, dim=1)
y_prob_without_furnace = y_prob_without_furnace.detach().cpu().numpy()

# Calculate the ROC curve and AUC
fpr_without_furnace, tpr_without_furnace, _ = roc_curve(labels.cpu().numpy(), y_prob_without_furnace[:, 1])
roc_auc_without_furnace = auc(fpr_without_furnace, tpr_without_furnace)

# Plotting the ROC curves
plt.figure()
plt.plot(fpr_with_furnace, tpr_with_furnace, color='darkorange',
         lw=2, label='ROC curve with Furnace Numbers (area = %0.2f)' % roc_auc_with_furnace)
plt.plot(fpr_without_furnace, tpr_without_furnace, color='darkgreen',
         lw=2, label='ROC curve without Furnace Numbers (area = %0.2f)' % roc_auc_without_furnace)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Visualize accuracy and loss
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].plot(train_loss, label='Train')
ax[0, 0].plot(val_loss, label='Val')
ax[0, 0].set_title('Loss')
ax[0, 0].legend()

ax[0, 1].plot(train_acc, label='Train')
ax[0, 1].plot(val_acc, label='Val')
ax[0, 1].set_title('Accuracy')
ax[0, 1].legend()

sns.heatmap(cm, annot=True, ax=ax[1, 0])
ax[1, 0].set_title('Confusion Matrix')

plt.show()