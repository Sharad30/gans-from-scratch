import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import Tuple, Optional
from loguru import logger


class BlackBlondDataset(Dataset):
    """Dataset for black and blond hair images derived from CelebA."""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 image_size: int = 64,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_root: Root directory containing 'black' and 'blond' folders
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to
            transform: Optional custom transforms
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # Set up paths
        self.black_dir = os.path.join(data_root, 'black')
        self.blond_dir = os.path.join(data_root, 'blond')
        
        # Verify directories exist
        if not os.path.exists(self.black_dir):
            raise ValueError(f"Black hair directory not found: {self.black_dir}")
        if not os.path.exists(self.blond_dir):
            raise ValueError(f"Blond hair directory not found: {self.blond_dir}")
        
        # Load and split data
        self.black_images, self.blond_images = self._load_and_split_data()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
        logger.info(f"Loaded {split} split: {len(self.black_images)} black, {len(self.blond_images)} blond images")
    
    def _load_and_split_data(self) -> Tuple[list, list]:
        """Load image paths and split into train/val/test."""
        # Get all image files
        black_files = [f for f in os.listdir(self.black_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        blond_files = [f for f in os.listdir(self.blond_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sort for reproducibility
        black_files.sort()
        blond_files.sort()
        
        # Create full paths
        black_paths = [os.path.join(self.black_dir, f) for f in black_files]
        blond_paths = [os.path.join(self.blond_dir, f) for f in blond_files]
        
        # Split data (70% train, 15% val, 15% test)
        def split_list(data_list):
            random.seed(42)  # For reproducibility
            random.shuffle(data_list)
            n = len(data_list)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            if self.split == 'train':
                return data_list[:train_end]
            elif self.split == 'val':
                return data_list[train_end:val_end]
            else:  # test
                return data_list[val_end:]
        
        return split_list(black_paths), split_list(blond_paths)
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transforms."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self) -> int:
        """Return length based on the larger domain."""
        return max(len(self.black_images), len(self.blond_images))
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample containing both black and blond images."""
        # Use modulo to handle different domain sizes
        black_idx = idx % len(self.black_images)
        blond_idx = idx % len(self.blond_images)
        
        # Load and transform images
        black_img = Image.open(self.black_images[black_idx]).convert('RGB')
        blond_img = Image.open(self.blond_images[blond_idx]).convert('RGB')
        
        black_tensor = self.transform(black_img)
        blond_tensor = self.transform(blond_img)
        
        return {
            'black': black_tensor,
            'blond': blond_tensor,
            'black_path': self.black_images[black_idx],
            'blond_path': self.blond_images[blond_idx]
        }


def create_dataloaders(data_root: str, 
                      batch_size: int = 16,
                      image_size: int = 64,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Create datasets
    train_dataset = BlackBlondDataset(data_root, 'train', image_size)
    val_dataset = BlackBlondDataset(data_root, 'val', image_size)
    test_dataset = BlackBlondDataset(data_root, 'test', image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader 