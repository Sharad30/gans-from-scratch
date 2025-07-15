import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from loguru import logger
import itertools


class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator."""
    
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """CycleGAN Generator using ResNet architecture."""
    
    def __init__(self, 
                 input_nc: int = 3,
                 output_nc: int = 3,
                 ngf: int = 64,
                 n_residual_blocks: int = 9):
        super(CycleGANGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """CycleGAN Discriminator using PatchGAN architecture."""
    
    def __init__(self, input_nc: int = 3, ndf: int = 64):
        super(CycleGANDiscriminator, self).__init__()
        
        # A bunch of convolutions one after another
        model = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # FCN classification layer
        model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]
        
        self.model = nn.Sequential(*model)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.model(x)


class ImagePool:
    """Buffer that stores previously generated images for discriminator training."""
    
    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        """Return images from the pool."""
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    random_id = int(torch.randint(0, self.pool_size, (1,)).item())
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGAN(nn.Module):
    """Complete CycleGAN model for bidirectional image translation."""
    
    def __init__(self,
                 input_nc: int = 3,
                 output_nc: int = 3,
                 ngf: int = 64,
                 ndf: int = 64,
                 n_residual_blocks: int = 9,
                 lambda_cycle: float = 10.0,
                 lambda_identity: float = 0.5,
                 device: str = 'cuda'):
        super(CycleGAN, self).__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.device = device
        
        # Create generators: G_AB (black->blond), G_BA (blond->black)
        self.G_AB = CycleGANGenerator(input_nc, output_nc, ngf, n_residual_blocks)
        self.G_BA = CycleGANGenerator(output_nc, input_nc, ngf, n_residual_blocks)
        
        # Create discriminators: D_A (black), D_B (blond)
        self.D_A = CycleGANDiscriminator(input_nc, ndf)
        self.D_B = CycleGANDiscriminator(output_nc, ndf)
        
        # Move to device
        self.G_AB.to(device)
        self.G_BA.to(device)
        self.D_A.to(device)
        self.D_B.to(device)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Image pools for storing previously generated images
        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()
        
        logger.info(f"CycleGAN initialized with λ_cycle={lambda_cycle}, λ_identity={lambda_identity}")
    
    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> Dict[str, float]:
        """Perform one training step."""
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        batch_size = real_A.size(0)
        
        # ==========================================
        # Train Generators
        # ==========================================
        self.optimizer_G.zero_grad()
        
        # Identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A) * self.lambda_identity
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B) * self.lambda_identity
        
        # GAN loss
        fake_B = self.G_AB(real_A)
        pred_fake = self.D_B(fake_B)
        target_real = torch.ones_like(pred_fake, requires_grad=False)
        loss_GAN_AB = self.criterion_GAN(pred_fake, target_real)
        
        fake_A = self.G_BA(real_B)
        pred_fake = self.D_A(fake_A)
        target_real = torch.ones_like(pred_fake, requires_grad=False)
        loss_GAN_BA = self.criterion_GAN(pred_fake, target_real)
        
        # Cycle loss
        recovered_A = self.G_BA(fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * self.lambda_cycle
        
        recovered_B = self.G_AB(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * self.lambda_cycle
        
        # Total generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_ABA + loss_cycle_BAB + loss_id_A + loss_id_B
        loss_G.backward()
        self.optimizer_G.step()
        
        # ==========================================
        # Train Discriminator A
        # ==========================================
        self.optimizer_D_A.zero_grad()
        
        # Real loss
        pred_real = self.D_A(real_A)
        target_real = torch.ones_like(pred_real, requires_grad=False)
        loss_D_A_real = self.criterion_GAN(pred_real, target_real)
        
        # Fake loss (with image pool)
        fake_A_pool = self.fake_A_pool.query(fake_A)
        pred_fake = self.D_A(fake_A_pool.detach())
        target_fake = torch.zeros_like(pred_fake, requires_grad=False)
        loss_D_A_fake = self.criterion_GAN(pred_fake, target_fake)
        
        # Total loss
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()
        
        # ==========================================
        # Train Discriminator B
        # ==========================================
        self.optimizer_D_B.zero_grad()
        
        # Real loss
        pred_real = self.D_B(real_B)
        target_real = torch.ones_like(pred_real, requires_grad=False)
        loss_D_B_real = self.criterion_GAN(pred_real, target_real)
        
        # Fake loss (with image pool)
        fake_B_pool = self.fake_B_pool.query(fake_B)
        pred_fake = self.D_B(fake_B_pool.detach())
        target_fake = torch.zeros_like(pred_fake, requires_grad=False)
        loss_D_B_fake = self.criterion_GAN(pred_fake, target_fake)
        
        # Total loss
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D_A': loss_D_A.item(),
            'loss_D_B': loss_D_B.item(),
            'loss_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(),
            'loss_identity': (loss_id_A + loss_id_B).item(),
            'loss_GAN': (loss_GAN_AB + loss_GAN_BA).item()
        }
    
    def generate_translations(self, real_A: torch.Tensor, real_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate translations for evaluation."""
        self.G_AB.eval()
        self.G_BA.eval()
        
        with torch.no_grad():
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            fake_B = self.G_AB(real_A)  # black -> blond
            fake_A = self.G_BA(real_B)  # blond -> black
            
            # Cycle consistency
            recovered_A = self.G_BA(fake_B)
            recovered_B = self.G_AB(fake_A)
        
        self.G_AB.train()
        self.G_BA.train()
        
        return {
            'real_A': real_A,
            'real_B': real_B,
            'fake_B': fake_B,
            'fake_A': fake_A,
            'recovered_A': recovered_A,
            'recovered_B': recovered_B
        } 