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
from sklearn.metrics import roc_curve, roc_auc_score
import csv

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
        self.crystal_count = {} 
        self.crystal_ids = []  

        classes = ['Pass', 'Fail']  # assuming these are the only two classes
        for i, class_ in enumerate(classes):
            class_dir = os.path.join(root_dir, class_)
            #print(f'Class directory: {class_dir}')  
            crystals = [entry.name for entry in os.scandir(class_dir) if not entry.is_symlink()]
            #print(f'Found crystals: {crystals}')  # And this line
            for crystal in crystals:
                if crystal not in self.crystal_count: 
                    self.crystal_count[crystal] = 0
                self.crystal_count[crystal] += 1
                #print(f'Processing crystal folder: {crystal} - Status: {class_}')
                furnace_num = int(crystal[0])  # extract first character as furnace number
                crystal_dir = os.path.join(class_dir, crystal)
                imgs = []
                for img_name in sorted(os.listdir(crystal_dir)):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(crystal_dir, img_name)
                        imgs.append(img_path)
                # Group every two images together
                for j in range(0, len(imgs), 3):
                    self.imgs.append([imgs[j], imgs[j + 1], imgs[j + 2]])
                    self.labels.append(i)  # assign label based on class
                    self.furnace_nums.append(furnace_num)
                    self.crystal_ids.append(crystal)  # store crystal ID
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img1 = self.transform(Image.open(self.imgs[index][0]))
        img2 = self.transform(Image.open(self.imgs[index][1]))
        img3 = self.transform(Image.open(self.imgs[index][2]))   # Load the third image
        img = torch.cat([img1, img2, img3], dim=0)  # concatenate along the channel dimension
        label = self.labels[index]
        furnace_num = torch.tensor([self.furnace_nums[index]], dtype=torch.long)  # pass as tensor
        crystal_id = self.crystal_ids[index]  # get the crystal ID
        return img, furnace_num, label, crystal_id  # return the crystal ID along with the other data

# Load the data
train_data = CustomImageFolder('/data/wesley/3_data_7_31_23/train_test_reflection/TX/train', transform=transform)
val_data = CustomImageFolder('/data/wesley/3_data_7_31_23/train_test_reflection/TX/val', transform=transform)
#test_data = CustomImageFolder('/data/wesley/3_data_7_31_23/train_test_reflection/TX/test', transform=transform)

# Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
#test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the model
model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)

# Change the first layer to accept the three-channel images
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Change the last layer to accept the furnace number along with the CNN features
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # adjust hidden layer size as required
    nn.ReLU(),
    nn.Linear(512, 2)
)

class CustomModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fc = nn.Linear(3, 2)  # one extra input for the furnace_num

    def forward(self, x, furnace_num):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, furnace_num], dim=1)
        x = self.fc(x)
        return x

model = CustomModel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
model = model.to(device)

# Define the loss function and the optimizer
# For binary classification
weights = [0.3, 0.7]  # class 0 is "Passes" and class 1 is "Fails"
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define the loss function and the optimizer/data/wesley/2_data/train_test/TX
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Define the path to the output directory
optimizer_name = "Adagrad"
weights_str = '-'.join(str(w) for w in weights)  # convert weights to a string
output_dir = f'/data/wesley/3_data_7_31_23/model_outputs_reflection/TX/{optimizer_name}_{weights_str}_{timestamp}'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Lists for saving epoch-wise losses and accuracies
train_losses, val_losses, train_accs, val_accs = [], [], [], []

best_acc = 0.0  # Track best validation accuracy
best_epoch = 0 # Track best epoch
val_acc = 0.0 

# Training loop
for epoch in range(10):  # 10 epochs, adjust as needed
    model.train()  
    train_loss = 0.0
    train_corrects = 0
    
    # Initialize empty lists for training set prediction probabilities
    train_pass_probs = []
    train_fail_probs = []
    
    # Initialize empty list to store training set predictions and their associated data
    train_epoch_preds = []
    
    for inputs, furnace_nums, labels, crystal_ids in train_loader:
        inputs, furnace_nums, labels = inputs.to(device), furnace_nums.float().to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, furnace_nums)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)

        with torch.no_grad():
            # Calculate prediction probabilities
            scores = nn.functional.softmax(outputs, dim=1).cpu().numpy()

        # Iterate over the scores and labels, appending the probabilities to the right list
        for score, label in zip(scores, labels.cpu().numpy()):
            if label == 0:  # class 'Pass'
                train_pass_probs.append(score[0])
            else:  # class 'Fail'
                train_fail_probs.append(score[1])
        
        # Code to save probabilities for each crystal in the epoch
        for j, score in enumerate(scores):
            crystal_id = crystal_ids[j]  # get the crystal ID
            human_label = 'Pass' if labels[j] == 0 else 'Fail'  # get the crystal label
            model_label = 'Pass' if preds[j] == 0 else 'Fail'
            model_probability = score[1]  # score for class 1 (Fail)
            if model_probability > 0.80: # change thresholds as necessary
                confidence = 'Confident Fail'
            elif model_probability < 0.20: # change threshold as necessary
                confidence = 'Confident Pass'
            else:
                confidence = 'Not confident'
            train_epoch_preds.append((crystal_id, human_label, model_label, model_probability, confidence))

    scheduler.step()

    model.eval()  # set model to evaluation mode
    val_scores = None
    val_labels = None
    val_loss = 0.0
    val_corrects = 0

    # Initialize empty lists for prediction probabilities
    pass_probs = []
    fail_probs = []
    
    # Initialize empty list to store predictions and their associated data
    epoch_preds = []
    
    with torch.no_grad():
        for i, (inputs, furnace_nums, labels, crystal_ids) in enumerate(val_loader):
            inputs, furnace_nums, labels = inputs.to(device), furnace_nums.float().to(device), labels.to(device)
            outputs = model(inputs, furnace_nums)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            scores = nn.functional.softmax(outputs, dim=1).cpu().numpy()  # calculate scores
            if val_scores is None:
                val_scores = scores
                val_labels = labels.cpu().numpy()
            else:
                val_scores = np.vstack([val_scores, scores])
                val_labels = np.hstack([val_labels, labels.cpu().numpy()])

            # Iterate over the scores and labels, appending the probabilities to the right list
            for score, label in zip(scores, labels.cpu().numpy()):
                if label == 0:  # class 'Pass'
                    pass_probs.append(score[0])
                else:  # class 'Fail'
                    fail_probs.append(score[1])
            
            # Code to save probabilities for each crystal in the epoch
            for j, score in enumerate(scores):
                crystal_id = crystal_ids[j]  # get the crystal ID
                human_label = 'Pass' if labels[j] == 0 else 'Fail' # get the crystal label
                model_label = 'Pass' if preds[j] == 0 else 'Fail'
                model_probability = score[1]  # score for class 1 (Fail)
                if model_probability > 0.80: # change thresholds as necessary
                    confidence = 'Confident Fail'
                elif model_probability < 0.20: # change threshold as necessary
                    confidence = 'Confident Pass'
                else:
                    confidence = 'Not confident'
                epoch_preds.append((crystal_id, human_label, model_label, model_probability, confidence))

    # Checking whether the dataset is loaded correctly - If this prints, something is wrong
    for crystal, count in val_data.crystal_count.items():
        if count != 1:
            print(f"Crystal {crystal} appears {count} times in the validation data.")

    # For the train epoch logs
    with open(os.path.join(output_dir, f'train_epoch_{epoch + 1}_log.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only once, i.e., when the file is created
        if epoch == 0:
            writer.writerow(['Crystal ID', 'Human label', 'Model label', 'Model Probability', 'Confidence'])
        for crystal_id, human_label, model_label, model_probability, confidence in train_epoch_preds:
            writer.writerow([crystal_id, human_label, model_label, f"{model_probability:.3f}", confidence])

    # For the general epoch logs
    with open(os.path.join(output_dir, f'epoch_{epoch + 1}_log.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only once, i.e., when the file is created
        if epoch == 0:
            writer.writerow(['Crystal ID', 'Human label', 'Model label', 'Model Probability', 'Confidence'])
        for crystal_id, human_label, model_label, model_probability, confidence in epoch_preds:
            writer.writerow([crystal_id, human_label, model_label, f"{model_probability:.3f}", confidence])
            
    # Calculate average losses and accuracies
    train_loss = train_loss / len(train_data)
    val_loss = val_loss / len(val_data)
    train_acc = train_corrects.double() / len(train_data)
    val_acc = val_corrects.double() / len(val_data)

    # Append to loss and accuracy lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f'Epoch {epoch+1}/{10}')
    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    plt.figure(figsize=(12, 6))
    plt.hist(train_pass_probs, bins=20, alpha=0.5, color='#009999',)
    plt.title('Training Pass Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'train_pass_histogram_epoch_{}.png'.format(epoch+1)))
    plt.close()

    # After the validation loop, plot the training set fail histogram
    plt.figure(figsize=(12, 6))
    plt.hist(train_fail_probs, bins=20, alpha=0.5, color='#ec6602',)
    plt.title('Training Fail Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'train_fail_histogram_epoch_{}.png'.format(epoch+1)))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.hist(pass_probs, bins=20, alpha=0.5, color='#009999',)
    plt.title('Pass Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'pass_histogram_epoch_{}.png'.format(epoch+1)))
    plt.close()

    # After the validation loop, plot the fail histogram
    plt.figure(figsize=(12, 6))
    plt.hist(fail_probs, bins=20, alpha=0.5, color='#ec6602',)
    plt.title('Fail Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, 'fail_histogram_epoch_{}.png'.format(epoch+1)))
    plt.close()

    # Deep copy the model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = model.state_dict()
        best_epoch = epoch + 1

# Load best model weights
model.load_state_dict(best_model_wts)

nb_classes = 2
confusion_matrix_best = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for inputs, furnace_nums, labels, crystal_ids in val_loader:
        inputs, furnace_nums, labels = inputs.to(device), furnace_nums.float().to(device), labels.to(device)
        outputs = model(inputs, furnace_nums)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix_best[t.long(), p.long()] += 1

# Save the model
torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))

# Confusion matrix
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for inputs, furnace_nums, labels, crystal_ids in val_loader:
        inputs, furnace_nums, labels = inputs.to(device), furnace_nums.float().to(device), labels.to(device)
        outputs = model(inputs, furnace_nums)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        

print(confusion_matrix)

# For the final epoch log
with open(os.path.join(output_dir, 'final_epoch_log.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Crystal ID', 'Human label', 'Model label', 'Model Probability', 'Confidence'])
    for crystal_id, human_label, model_label, model_probability, confidence in epoch_preds:
        writer.writerow([crystal_id, human_label, model_label, f"{model_probability:.3f}", confidence])

# After the training loop, plot the pass histogram
plt.figure(figsize=(12, 6))
plt.hist(pass_probs, bins=20, alpha=0.5, color='#009999', label='Pass')
plt.title('Pass Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, 'final_pass_histogram.png'))
plt.close()

# After the training loop, plot the fail histogram
plt.figure(figsize=(12, 6))
plt.hist(fail_probs, bins=20, alpha=0.5, color='#ec6602', label='Fail')
plt.title('Fail Prediction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, 'final_fail_histogram.png'))
plt.close()

print('Training complete. Best val Acc: {:4f}'.format(best_acc))

# Delete all non-best model log files
#for file in os.listdir(output_dir):
#    if file.endswith('_log.txt'):
#        os.remove(os.path.join(output_dir, file))

# Plot and save the confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix', size = 15)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

#save best confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix_best, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title(f'Confusion Matrix for Best Epoch ({best_epoch})', size = 15)
plt.savefig(os.path.join(output_dir, f'confusion_matrix_best_{best_epoch}.png'))

# calculate TPR, FPR, and AUC-ROC
fpr, tpr, _ = roc_curve(val_labels, val_scores[:, 1], pos_label=1)  # 1 is the positive class
roc_auc = roc_auc_score(val_labels, val_scores[:, 1])

# Plotting ROC curve
plt.figure()
lw = 2  # line width
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, 'roc_plot.png'))  # save plot
plt.show()

#if isinstance(train_accs, torch.Tensor):
#    train_accs = train_accs.cpu().numpy()

# Plotting the training and validation accuracy
#plt.figure(figsize=(10, 5))
#plt.title("Training and Validation Accuracy")
#plt.plot(train_accs, label='Training Accuracy')
#plt.plot(val_accs.cpu().numpy(), label='Validation Accuracy')
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
#plt.legend()
#plt.savefig(os.path.join(output_dir, 'acc_plot.png'))

