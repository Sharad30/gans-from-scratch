import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from typing import Union, List, Optional


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Input tensor normalized to [-1, 1]
    
    Returns:
        Tensor in [0, 1] range
    """
    return (tensor + 1.0) / 2.0


def save_image_grid(images: torch.Tensor, 
                   save_path: str, 
                   nrow: int = 8, 
                   padding: int = 2,
                   normalize: bool = False) -> None:
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images [N, C, H, W]
        save_path: Path to save the image
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize images
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(images, save_path, nrow=nrow, padding=padding, normalize=normalize)


def create_comparison_grid(real_images: torch.Tensor,
                          fake_images: torch.Tensor,
                          save_path: str,
                          titles: Optional[List[str]] = None) -> None:
    """
    Create a comparison grid showing real vs fake images.
    
    Args:
        real_images: Real images tensor
        fake_images: Generated images tensor  
        save_path: Path to save the comparison
        titles: Optional titles for the images
    """
    # Ensure same batch size
    batch_size = min(real_images.size(0), fake_images.size(0))
    real_images = real_images[:batch_size]
    fake_images = fake_images[:batch_size]
    
    # Denormalize if needed
    if real_images.min() < 0:
        real_images = denormalize_tensor(real_images)
    if fake_images.min() < 0:
        fake_images = denormalize_tensor(fake_images)
    
    # Create alternating grid (real, fake, real, fake, ...)
    comparison = torch.stack([real_images, fake_images], dim=1)  # [B, 2, C, H, W]
    comparison = comparison.view(-1, *real_images.shape[1:])     # [B*2, C, H, W]
    
    save_image_grid(comparison, save_path, nrow=2, padding=2)


def plot_training_curves(losses: dict, save_path: str) -> None:
    """
    Plot training loss curves.
    
    Args:
        losses: Dictionary containing loss histories
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for loss_name, loss_values in losses.items():
        plt.plot(loss_values, label=loss_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_cyclegan_results(results: dict, save_path: str, num_samples: int = 4) -> None:
    """
    Save CycleGAN translation results in a structured grid.
    
    Args:
        results: Dictionary containing translation results
        save_path: Path to save the results
        num_samples: Number of samples to visualize
    """
    # Extract images (limit to num_samples)
    real_A = results['real_A'][:num_samples]
    fake_B = results['fake_B'][:num_samples] 
    recovered_A = results['recovered_A'][:num_samples]
    real_B = results['real_B'][:num_samples]
    fake_A = results['fake_A'][:num_samples]
    recovered_B = results['recovered_B'][:num_samples]
    
    # Denormalize
    images = [real_A, fake_B, recovered_A, real_B, fake_A, recovered_B]
    images = [denormalize_tensor(img) for img in images]
    
    # Create grid: each row shows one sample's transformation
    # Columns: Real_A -> Fake_B -> Recovered_A | Real_B -> Fake_A -> Recovered_B
    rows = []
    for i in range(num_samples):
        row = torch.cat([
            images[0][i:i+1],  # Real_A
            images[1][i:i+1],  # Fake_B  
            images[2][i:i+1],  # Recovered_A
            torch.zeros_like(images[0][i:i+1]),  # Separator
            images[3][i:i+1],  # Real_B
            images[4][i:i+1],  # Fake_A
            images[5][i:i+1]   # Recovered_B
        ], dim=0)
        rows.append(row)
    
    final_grid = torch.cat(rows, dim=0)
    save_image_grid(final_grid, save_path, nrow=7, padding=2)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image.
    
    Args:
        tensor: Image tensor [C, H, W] in range [0, 1]
    
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batch
    
    # Ensure range [0, 1]
    if tensor.min() < 0:
        tensor = denormalize_tensor(tensor)
    
    # Convert to numpy
    tensor_np = tensor.permute(1, 2, 0).cpu().numpy()
    tensor_np = (tensor_np * 255).astype(np.uint8)
    
    return Image.fromarray(tensor_np)


def create_side_by_side_comparison(image1: torch.Tensor,
                                  image2: torch.Tensor, 
                                  save_path: str,
                                  title1: str = "Image 1",
                                  title2: str = "Image 2") -> None:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        image1: First image tensor
        image2: Second image tensor
        save_path: Path to save comparison
        title1: Title for first image
        title2: Title for second image
    """
    # Convert to PIL
    pil1 = tensor_to_pil(image1)
    pil2 = tensor_to_pil(image2)
    
    # Create side-by-side comparison
    width, height = pil1.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(pil1, (0, 0))
    comparison.paste(pil2, (width, 0))
    
    # Add titles using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(comparison)
    ax.text(width//2, -20, title1, ha='center', fontsize=12, weight='bold')
    ax.text(width + width//2, -20, title2, ha='center', fontsize=12, weight='bold')
    ax.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_latent_interpolation(model, 
                                  start_noise: torch.Tensor,
                                  end_noise: torch.Tensor,
                                  save_path: str,
                                  num_steps: int = 8) -> None:
    """
    Visualize interpolation in latent space for DCGAN.
    
    Args:
        model: DCGAN model
        start_noise: Starting noise vector
        end_noise: Ending noise vector
        save_path: Path to save interpolation
        num_steps: Number of interpolation steps
    """
    model.eval()
    interpolation_images = []
    
    with torch.no_grad():
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated_noise = (1 - alpha) * start_noise + alpha * end_noise
            generated_image = model.generator(interpolated_noise)
            interpolation_images.append(generated_image)
    
    # Create grid
    interpolation_tensor = torch.cat(interpolation_images, dim=0)
    save_image_grid(
        denormalize_tensor(interpolation_tensor), 
        save_path, 
        nrow=num_steps, 
        padding=2
    )
    
    model.train() 


def visualize_dataset_samples(dir_a: str, 
                            dir_b: str,
                            num_samples: int = 8,
                            figsize: tuple = (14.24, 4.36),
                            dpi: int = 100,
                            save_path: Optional[str] = None,
                            titles: Optional[List[str]] = None,
                            random_seed: int = 42) -> None:
    """
    Visualize sample images from two directories in a grid format.
    
    Args:
        dir_a: Path to first directory (e.g., black hair images)
        dir_b: Path to second directory (e.g., blond hair images)
        num_samples: Number of samples to show from each directory
        figsize: Figure size (width, height)
        dpi: DPI for the figure
        save_path: Optional path to save the visualization
        titles: Optional titles for the two rows
        random_seed: Random seed for reproducible sampling
    """
    # Get image lists
    imgs_a = [f for f in os.listdir(dir_a) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    imgs_b = [f for f in os.listdir(dir_b) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(imgs_a) < num_samples:
        print(f"Warning: Only {len(imgs_a)} images found in {dir_a}, requested {num_samples}")
        num_samples = min(num_samples, len(imgs_a))
    
    if len(imgs_b) < num_samples:
        print(f"Warning: Only {len(imgs_b)} images found in {dir_b}, requested {num_samples}")
        num_samples = min(num_samples, len(imgs_b))
    
    # Sample images randomly
    random.seed(random_seed)
    samples_a = random.sample(imgs_a, num_samples)
    random.seed(random_seed)
    samples_b = random.sample(imgs_b, num_samples)
    
    # Create figure
    fig = plt.figure(dpi=dpi, figsize=figsize)
    
    # Plot images from directory A (top row)
    for i in range(num_samples):
        ax = plt.subplot(2, num_samples, i + 1)
        img_path = os.path.join(dir_a, samples_a[i])
        img = Image.open(img_path)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        # Add title only to the first image of the top row
        if titles and i == 0:
            plt.title(titles[0], fontsize=12, weight='bold', pad=10)
    
    # Plot images from directory B (bottom row)
    for i in range(num_samples):
        ax = plt.subplot(2, num_samples, num_samples + i + 1)
        img_path = os.path.join(dir_b, samples_b[i])
        img = Image.open(img_path)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        # Add title only to the first image of the bottom row
        if titles and i == 0:
            plt.title(titles[1], fontsize=12, weight='bold', pad=10)
    
    # Adjust layout with more space between rows
    plt.subplots_adjust(wspace=0.01, hspace=0.25)
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    plt.show()


def visualize_single_dataset_samples(directory: str,
                                   num_samples: int = 8,
                                   nrow: int = 4,
                                   figsize: tuple = (12, 6),
                                   dpi: int = 100,
                                   save_path: Optional[str] = None,
                                   title: Optional[str] = None,
                                   random_seed: int = 42) -> None:
    """
    Visualize sample images from a single directory in a grid format.
    
    Args:
        directory: Path to image directory
        num_samples: Number of samples to show
        nrow: Number of images per row
        figsize: Figure size (width, height)
        dpi: DPI for the figure
        save_path: Optional path to save the visualization
        title: Optional title for the visualization
        random_seed: Random seed for reproducible sampling
    """
    # Get image list
    imgs = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(imgs) < num_samples:
        print(f"Warning: Only {len(imgs)} images found in {directory}, requested {num_samples}")
        num_samples = min(num_samples, len(imgs))
    
    # Sample images randomly
    random.seed(random_seed)
    samples = random.sample(imgs, num_samples)
    
    # Calculate grid dimensions
    ncols = nrow
    nrows = (num_samples + ncols - 1) // ncols
    
    # Create figure
    fig = plt.figure(dpi=dpi, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=14, weight='bold')
    
    # Plot images
    for i, img_name in enumerate(samples):
        ax = plt.subplot(nrows, ncols, i + 1)
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    
    # Hide empty subplots
    for i in range(num_samples, nrows * ncols):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    plt.show() 