import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from loguru import logger


class DCGANGenerator(nn.Module):
    """DCGAN Generator for generating images from noise."""
    
    def __init__(self, 
                 nz: int = 100,           # Size of latent vector
                 ngf: int = 64,           # Generator feature map size
                 nc: int = 3,             # Number of channels
                 image_size: int = 64):   # Output image size
        super(DCGANGenerator, self).__init__()
        
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.image_size = image_size
        
        # Calculate the initial spatial size
        # For 64x64 output, we need 4x4 initial size with 4 upsampling layers
        self.init_size = image_size // 16  # 4 for 64x64
        
        # Main generator network
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            # State size: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, self.init_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (nc) x 64 x 64
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights according to DCGAN paper."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, input):
        """Forward pass through generator."""
        return self.main(input)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator for distinguishing real from fake images."""
    
    def __init__(self, 
                 nc: int = 3,             # Number of channels
                 ndf: int = 64,           # Discriminator feature map size
                 image_size: int = 64):   # Input image size
        super(DCGANDiscriminator, self).__init__()
        
        self.nc = nc
        self.ndf = ndf
        self.image_size = image_size
        
        # Main discriminator network
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # State size: 1 x 1 x 1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights according to DCGAN paper."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, input):
        """Forward pass through discriminator."""
        return self.main(input).view(-1, 1).squeeze(1)


class DCGAN(nn.Module):
    """Complete DCGAN model combining generator and discriminator."""
    
    def __init__(self,
                 nz: int = 100,
                 ngf: int = 64,
                 ndf: int = 64,
                 nc: int = 3,
                 image_size: int = 64,
                 device: str = 'cuda'):
        super(DCGAN, self).__init__()
        
        self.nz = nz
        self.device = device
        
        # Create generator and discriminator
        self.generator = DCGANGenerator(nz, ngf, nc, image_size)
        self.discriminator = DCGANDiscriminator(nc, ndf, image_size)
        
        # Move to device
        self.generator.to(device)
        self.discriminator.to(device)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        logger.info(f"DCGAN initialized with nz={nz}, ngf={ngf}, ndf={ndf}")
    
    def generate_noise(self, batch_size: int) -> torch.Tensor:
        """Generate random noise vector."""
        return torch.randn(batch_size, self.nz, 1, 1, device=self.device)
    
    def train_step(self, real_images: torch.Tensor) -> Tuple[float, float]:
        """Perform one training step."""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels
        real_label = torch.ones(batch_size, device=self.device)
        fake_label = torch.zeros(batch_size, device=self.device)
        
        # ==========================================
        # Train Discriminator
        # ==========================================
        self.discriminator.zero_grad()
        
        # Train with real images
        output_real = self.discriminator(real_images)
        loss_d_real = self.criterion(output_real, real_label)
        
        # Train with fake images
        noise = self.generate_noise(batch_size)
        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        loss_d_fake = self.criterion(output_fake, fake_label)
        
        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()
        
        # ==========================================
        # Train Generator
        # ==========================================
        self.generator.zero_grad()
        
        # Generate fake images and try to fool discriminator
        output_fake = self.discriminator(fake_images)
        loss_g = self.criterion(output_fake, real_label)  # Want discriminator to think fake is real
        
        loss_g.backward()
        self.optimizer_g.step()
        
        return loss_d.item(), loss_g.item()
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate sample images."""
        self.generator.eval()
        with torch.no_grad():
            noise = self.generate_noise(num_samples)
            samples = self.generator(noise)
        self.generator.train()
        return samples 