import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST  # Changed from CIFAR10
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 64
IMAGE_SIZE = 28
LATENT_DIM = 100
NUM_CLASSES = 10  # FashionMNIST also has 10 classes
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_WORKERS = 4
MAX_EPOCHS = 100
# Configuration for image saving
SAVE_IMAGES = True
IMAGE_SAVE_PATH = "./CGAN_Fashion/generated_images"  # Updated path
SAVE_FREQUENCY = 5


class GANDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                 num_workers=NUM_WORKERS):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        
    def prepare_data(self):
        # Download the dataset
        FashionMNIST(root='./data', train=True, download=True)
    
    def setup(self, stage=None):
        # Define transforms for grayscale images
        transform = transforms.Compose([
            # transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Single channel normalization
        ])
        
        # Create datasets
        self.train_dataset = FashionMNIST(root='./data', train=True, 
                                     transform=transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Input processing
        self.input_layer = nn.Linear(latent_dim + num_classes, 7 * 7 * 256)
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 7x7x256 -> 14x14x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # TODO: więcej warstw

            # 14x14x128 -> 28x28x1 (final output layer now produces 28x28 images)
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Process labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embeddings
        z = torch.cat([z, label_embedding], dim=1)
        
        # Process and reshape input
        x = self.input_layer(z)
        x = x.view(-1, 256, 7, 7)
        
        # Generate image
        x = self.conv_layers(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional layers - adjusted for grayscale input at 28x28
        self.conv_layers = nn.Sequential(
            # 28x28x1 -> 14x14x64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14x64 -> 7x7x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Embedding layer for class condition
        self.label_embedding = nn.Embedding(num_classes, 4*4*512)
        
        # Output layer
        self.output_layer = nn.Linear(2*4*4*512, 1)
        
    def forward(self, img, labels):
        # Process image
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        
        # Process labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate and classify
        x = torch.cat([x, label_embedding], dim=1)
        x = self.output_layer(x)
        return x


class CGAN(pl.LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, lr=LR, beta1=BETA1, beta2=BETA2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Initialize models
        self.generator = Generator(latent_dim, num_classes)
        self.discriminator = Discriminator(num_classes)
        
        # For logging
        self.validation_z = torch.randn(8, latent_dim)
        self.validation_labels = torch.arange(0, 8) % num_classes
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Set manual optimization for multiple optimizers
        self.automatic_optimization = False
    
    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        real_imgs, real_labels = batch
        batch_size = real_imgs.size(0)
        
        # Sample noise and random labels
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        # Ground truth labels
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)
        
        # Train Discriminator
        opt_d.zero_grad()
        
        # Loss for real images with real labels
        validity_real = self.discriminator(real_imgs, real_labels)
        d_real_loss = self.adversarial_loss(validity_real, valid)
        
        # Loss for fake images with fake labels
        fake_imgs = self.generator(z, fake_labels)
        validity_fake = self.discriminator(fake_imgs.detach(), fake_labels)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)
        
        # Loss for real images with incorrect labels
        shuffled_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        validity_wrong = self.discriminator(real_imgs, shuffled_labels)
        d_wrong_loss = self.adversarial_loss(validity_wrong, fake)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss + d_wrong_loss) / 3
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Train Generator
        opt_g.zero_grad()
        
        # Generate fake images
        fake_imgs = self.generator(z, fake_labels)
        
        # Calculate adversarial loss
        validity = self.discriminator(fake_imgs, fake_labels)
        g_loss = self.adversarial_loss(validity, valid)
        
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Log metrics
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('d_loss', d_loss, prog_bar=True)
        
        return {"g_loss": g_loss, "d_loss": d_loss}
    
    def configure_optimizers(self):
        opt_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return [opt_g, opt_d], []
    
    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        labels = self.validation_labels.to(self.device)
        
        # Generate images
        with torch.inference_mode():
            fake_imgs = self.generator(z, labels)
        
        # Log images - normalized differently for grayscale
        grid = torchvision.utils.make_grid(fake_imgs, normalize=True, value_range=(-1, 1))
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        # Save images to disk if enabled
        if SAVE_IMAGES and (self.current_epoch % SAVE_FREQUENCY == 0 or self.current_epoch == self.trainer.max_epochs - 1):
            epoch_save_path = os.path.join(IMAGE_SAVE_PATH, f"epoch_{self.current_epoch}")
            generate_and_show(self,
                              num_images=8,
                              device=self.device,
                              save_path=epoch_save_path, filename="samples.png")


# Visualization functions
def show_images(images, num_images=8, cols=4, save_path=None, filename=None):
    """Display or save a batch of images."""
    fig = plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(num_images//cols + 1, cols, i + 1)
        # Handle both grayscale (1 channel) and RGB (3 channels)
        img = images[i].cpu().detach()
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0).numpy()  # Remove channel dimension
            img = (img * 0.5) + 0.5  # Denormalize
            ax.imshow(img, cmap='gray')
        else:  # RGB
            img = img.numpy().transpose(1, 2, 0)
            img = (img * 0.5) + 0.5  # Denormalize
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename) if filename else os.path.join(save_path, "generated.png")
        plt.savefig(full_path)
        plt.close(fig)
    else:
        plt.show()


def generate_and_show(model, num_images=8, device='cuda', save_path=None, filename=None):
    """Generate and display/save images from the model."""
    model.eval()
    with torch.inference_mode():
        z = torch.randn(num_images, model.latent_dim, device=device)
        labels = torch.arange(0, num_images, device=device) % model.num_classes
        images = model(z, labels)
    show_images(images, num_images=num_images, save_path=save_path, filename=filename)
    return images


# Main function
def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Initialize data module
    data_module = GANDataModule()
    
    # Initialize model
    model = CGAN()
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='g_loss',
        dirpath='./CGAN_Fashion/checkpoints',  # Updated path
        filename='cgan-fashion-{epoch:02d}',
        save_top_k=3,
        mode='min',
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save model
    trainer.save_checkpoint("cgan_fashion_final.ckpt")


if __name__ == "__main__":
    main()