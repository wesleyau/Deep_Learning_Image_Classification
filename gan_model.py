import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_shape),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Hyperparameters
latent_dim = 100
img_size = 64
batch_size = 64
num_epochs = 100
learning_rate = 0.0002

# Initialize generator and discriminator
generator = Generator(latent_dim, img_size * img_size)
discriminator = Discriminator(img_size * img_size)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Dataset and dataloader
dataset = ImageFolder(root='path/to/data', transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Adversarial ground truth
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # Train generator
        optimizer_G.zero_grad()
        
        # Generate a batch of noise samples
        z = torch.randn(batch_size, latent_dim).to(device)
        generated_images = generator(z)
        
        # Generator loss
        g_loss = adversarial_loss(discriminator(generated_images), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Train discriminator
        optimizer_D.zero_grad()
        
        # Discriminator loss on real images
        real_loss = adversarial_loss(discriminator(real_images), valid)
        
        # Discriminator loss on generated images
        fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
        
        d_loss = 0.5 * (real_loss + fake_loss)
        
        d_loss.backward()
        optimizer_D.step()
        
        # Print progress
        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# Save the trained models
torch.save(generator.state_dict(), "generator_model.pth")
torch.save(discriminator.state_dict(), "discriminator_model.pth")
