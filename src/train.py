import argparse
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import time
from datetime import datetime
import json
from typing import Dict, Any

from datasets import create_dataloaders
from models import DCGAN, CycleGAN
from utils.visualization import save_image_grid, denormalize_tensor
from utils.logging_utils import setup_logging, log_losses


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCGAN or CycleGAN on Black-Blond dataset')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['dcgan', 'cyclegan'], required=True,
                       help='Model to train: dcgan or cyclegan')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing black and blond folders')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of images (will be resized to this)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 for Adam optimizer')
    
    # Model architecture parameters
    parser.add_argument('--nz', type=int, default=100,
                       help='Size of latent vector (DCGAN only)')
    parser.add_argument('--ngf', type=int, default=64,
                       help='Generator feature map size')
    parser.add_argument('--ndf', type=int, default=64,
                       help='Discriminator feature map size')
    
    # CycleGAN specific parameters
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                       help='Weight for cycle consistency loss (CycleGAN only)')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                       help='Weight for identity loss (CycleGAN only)')
    parser.add_argument('--n_residual_blocks', type=int, default=9,
                       help='Number of residual blocks in generator (CycleGAN only)')
    
    # Logging and checkpointing
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save model every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                       help='Log training stats every N iterations')
    parser.add_argument('--sample_freq', type=int, default=500,
                       help='Generate samples every N iterations')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu')
    
    return parser.parse_args()


def create_model(args) -> nn.Module:
    """Create the specified model."""
    if args.model == 'dcgan':
        model = DCGAN(
            nz=args.nz,
            ngf=args.ngf,
            ndf=args.ndf,
            nc=3,
            image_size=args.image_size,
            device=args.device
        )
    elif args.model == 'cyclegan':
        model = CycleGAN(
            input_nc=3,
            output_nc=3,
            ngf=args.ngf,
            ndf=args.ndf,
            n_residual_blocks=args.n_residual_blocks,
            lambda_cycle=args.lambda_cycle,
            lambda_identity=args.lambda_identity,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def create_experiment_name(data_root: str, model: str, batch_size: int, epochs: int) -> str:
    """Create experiment name following the convention: dataset_model_batch_size_epochs"""
    # Extract dataset name from data_root path
    dataset_name = os.path.basename(os.path.normpath(data_root))
    if not dataset_name or dataset_name == '.':
        dataset_name = 'data'
    
    return f"{dataset_name}_{model}_{batch_size}_{epochs}"


def setup_experiment_directory(experiment_name: str) -> str:
    """Setup experiment directory structure."""
    base_dir = os.path.join(os.getcwd(), 'experiments', experiment_name)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'samples', 'tensorboard']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    return base_dir


def save_checkpoint(model: nn.Module, epoch: int, experiment_dir: str, args: argparse.Namespace):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    save_path = os.path.join(checkpoint_dir, f'{args.model}_epoch_{epoch}.pth')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'experiment_name': os.path.basename(experiment_dir)
    }
    
    if hasattr(model, 'optimizer_g'):
        checkpoint['optimizer_g_state_dict'] = model.optimizer_g.state_dict()
    if hasattr(model, 'optimizer_d'):
        checkpoint['optimizer_d_state_dict'] = model.optimizer_d.state_dict()
    if hasattr(model, 'optimizer_G'):
        checkpoint['optimizer_G_state_dict'] = model.optimizer_G.state_dict()
    if hasattr(model, 'optimizer_D_A'):
        checkpoint['optimizer_D_A_state_dict'] = model.optimizer_D_A.state_dict()
    if hasattr(model, 'optimizer_D_B'):
        checkpoint['optimizer_D_B_state_dict'] = model.optimizer_D_B.state_dict()
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> int:
    """Load model checkpoint and return epoch number."""
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if hasattr(model, 'optimizer_g') and 'optimizer_g_state_dict' in checkpoint:
        model.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    if hasattr(model, 'optimizer_d') and 'optimizer_d_state_dict' in checkpoint:
        model.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    if hasattr(model, 'optimizer_G') and 'optimizer_G_state_dict' in checkpoint:
        model.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if hasattr(model, 'optimizer_D_A') and 'optimizer_D_A_state_dict' in checkpoint:
        model.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
    if hasattr(model, 'optimizer_D_B') and 'optimizer_D_B_state_dict' in checkpoint:
        model.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
    
    epoch = checkpoint['epoch']
    logger.info(f"Checkpoint loaded from epoch {epoch}")
    return epoch


def train_dcgan(model: DCGAN, train_loader, val_loader, args, writer: SummaryWriter, experiment_dir: str):
    """Train DCGAN model."""
    logger.info("Starting DCGAN training...")
    
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, args.resume)
    
    global_step = start_epoch * len(train_loader)
    samples_dir = os.path.join(experiment_dir, 'samples')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses_d = []
        epoch_losses_g = []
        
        for i, batch in enumerate(train_loader):
            # Use only one domain for DCGAN (you can choose black or blond)
            real_images = batch['black']  # or batch['blond']
            
            # Train step
            loss_d, loss_g = model.train_step(real_images)
            
            epoch_losses_d.append(loss_d)
            epoch_losses_g.append(loss_g)
            
            # Logging
            if i % args.log_freq == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(train_loader)}] "
                          f"Loss_D: {loss_d:.4f} Loss_G: {loss_g:.4f}")
                
                writer.add_scalar('Train/Loss_D', loss_d, global_step)
                writer.add_scalar('Train/Loss_G', loss_g, global_step)
            
            # Generate samples
            if i % args.sample_freq == 0:
                samples = model.generate_samples(16)
                save_image_grid(
                    denormalize_tensor(samples), 
                    os.path.join(samples_dir, f'samples_epoch_{epoch}_batch_{i}.png'),
                    nrow=4
                )
                
                # Log to tensorboard
                writer.add_images('Generated_Samples', denormalize_tensor(samples), global_step)
            
            global_step += 1
        
        # Epoch logging
        avg_loss_d = sum(epoch_losses_d) / len(epoch_losses_d)
        avg_loss_g = sum(epoch_losses_g) / len(epoch_losses_g)
        
        logger.info(f"Epoch [{epoch}/{args.epochs}] completed - "
                   f"Avg Loss_D: {avg_loss_d:.4f} Avg Loss_G: {avg_loss_g:.4f}")
        
        writer.add_scalar('Epoch/Loss_D', avg_loss_d, epoch)
        writer.add_scalar('Epoch/Loss_G', avg_loss_g, epoch)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, epoch, experiment_dir, args)


def train_cyclegan(model: CycleGAN, train_loader, val_loader, args, writer: SummaryWriter, experiment_dir: str):
    """Train CycleGAN model."""
    logger.info("Starting CycleGAN training...")
    
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, args.resume)
    
    global_step = start_epoch * len(train_loader)
    samples_dir = os.path.join(experiment_dir, 'samples')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {
            'loss_G': [],
            'loss_D_A': [],
            'loss_D_B': [],
            'loss_cycle': [],
            'loss_identity': [],
            'loss_GAN': []
        }
        
        for i, batch in enumerate(train_loader):
            real_A = batch['black']   # Domain A: black hair
            real_B = batch['blond']   # Domain B: blond hair
            
            # Train step
            losses = model.train_step(real_A, real_B)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key].append(value)
            
            # Logging
            if i % args.log_freq == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(train_loader)}] "
                          f"Loss_G: {losses['loss_G']:.4f} Loss_D_A: {losses['loss_D_A']:.4f} "
                          f"Loss_D_B: {losses['loss_D_B']:.4f} Loss_Cycle: {losses['loss_cycle']:.4f}")
                
                for key, value in losses.items():
                    writer.add_scalar(f'Train/{key}', value, global_step)
            
            # Generate samples
            if i % args.sample_freq == 0:
                with torch.no_grad():
                    # Take a small batch for visualization
                    sample_A = real_A[:4]
                    sample_B = real_B[:4]
                    translations = model.generate_translations(sample_A, sample_B)
                    
                    # Save translation results
                    save_translation_grid(
                        translations,
                        os.path.join(samples_dir, f'translations_epoch_{epoch}_batch_{i}.png')
                    )
                    
                    # Log to tensorboard
                    log_translations_to_tensorboard(writer, translations, global_step)
            
            global_step += 1
        
        # Epoch logging
        avg_losses = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
        
        logger.info(f"Epoch [{epoch}/{args.epochs}] completed - " +
                   " ".join([f"{key}: {value:.4f}" for key, value in avg_losses.items()]))
        
        for key, value in avg_losses.items():
            writer.add_scalar(f'Epoch/{key}', value, epoch)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, epoch, experiment_dir, args)


def save_translation_grid(translations: Dict[str, torch.Tensor], save_path: str):
    """Save CycleGAN translation results as a grid."""
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Create a grid showing: Real_A, Fake_B, Recovered_A, Real_B, Fake_A, Recovered_B
    images = [
        translations['real_A'],
        translations['fake_B'], 
        translations['recovered_A'],
        translations['real_B'],
        translations['fake_A'],
        translations['recovered_B']
    ]
    
    # Denormalize and create grid
    denorm_images = [denormalize_tensor(img) for img in images]
    grid = make_grid(torch.cat(denorm_images, 0), nrow=len(denorm_images[0]), padding=2, normalize=False)
    
    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_pil = Image.fromarray((grid_np * 255).astype('uint8'))
    grid_pil.save(save_path)


def log_translations_to_tensorboard(writer: SummaryWriter, translations: Dict[str, torch.Tensor], step: int):
    """Log translation results to TensorBoard."""
    for key, images in translations.items():
        writer.add_images(f'Translations/{key}', denormalize_tensor(images), step)


def main():
    args = get_args()
    
    # Create experiment name and directory
    experiment_name = create_experiment_name(args.data_root, args.model, args.batch_size, args.epochs)
    experiment_dir = setup_experiment_directory(experiment_name)
    
    # Setup logging
    setup_logging(os.path.join(experiment_dir, 'training.log'))
    
    # Save configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    config_data = vars(args).copy()
    config_data['experiment_name'] = experiment_name
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = str(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_root, 
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = create_model(args)
    
    # Setup TensorBoard
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    
    # Start training
    logger.info(f"Starting training with {args.epochs} epochs...")
    start_time = time.time()
    
    epoch = 0  # Initialize epoch for error handling
    try:
        if args.model == 'dcgan':
            train_dcgan(model, train_loader, val_loader, args, writer, experiment_dir)
        elif args.model == 'cyclegan':
            train_cyclegan(model, train_loader, val_loader, args, writer, experiment_dir)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        checkpoint_path = os.path.join(experiment_dir, 'checkpoints', f'{args.model}_interrupted.pth')
        save_checkpoint(model, epoch, experiment_dir, args)
    
    finally:
        writer.close()


if __name__ == '__main__':
    main()
