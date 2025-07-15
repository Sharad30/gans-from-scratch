import argparse
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from loguru import logger
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt

from datasets import create_dataloaders
from models import DCGAN, CycleGAN
from utils.visualization import (
    save_image_grid, denormalize_tensor, create_comparison_grid,
    save_cyclegan_results, create_side_by_side_comparison,
    visualize_latent_interpolation, tensor_to_pil
)
from utils.logging_utils import setup_logging


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DCGAN or CycleGAN models')
    
    # Model and checkpoint
    parser.add_argument('--model', type=str, choices=['dcgan', 'cyclegan'], required=True,
                       help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data parameters - make data_root optional for single image prediction
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory containing black and blond folders')
    parser.add_argument('--input_path', type=str, default=None,
                       help='Path to single image or folder for prediction (CycleGAN only)')
    parser.add_argument('--direction', type=str, choices=['AtoB', 'BtoA', 'both'], default='both',
                       help='Translation direction: AtoB (black->blond), BtoA (blond->black), or both')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of images')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate/evaluate')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results (auto-detected if not specified)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu')
    
    # Evaluation modes
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate samples from model')
    parser.add_argument('--test_translations', action='store_true',
                       help='Test translations (CycleGAN only)')
    parser.add_argument('--latent_interpolation', action='store_true',
                       help='Perform latent interpolation (DCGAN only)')
    parser.add_argument('--side_by_side', action='store_true',
                       help='Create side-by-side comparisons')
    parser.add_argument('--all', action='store_true',
                       help='Run all evaluation modes')
    parser.add_argument('--predict', action='store_true',
                       help='Predict on single image or folder (requires --input_path)')
    parser.add_argument('--batch_quality', action='store_true',
                       help='Use batch-quality prediction (matches translation batch process)')
    
    return parser.parse_args()


def determine_evaluation_directory(checkpoint_path: str, save_dir: Optional[str] = None) -> str:
    """Determine where to save evaluation results."""
    if save_dir:
        return save_dir
    
    # Try to auto-detect experiment directory from checkpoint path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir.endswith('checkpoints'):
        # This is likely an experiment directory
        experiment_dir = os.path.dirname(checkpoint_dir)
        evaluation_dir = os.path.join(experiment_dir, 'evaluation')
        os.makedirs(evaluation_dir, exist_ok=True)
        return evaluation_dir
    
    # Fallback to creating evaluation directory next to checkpoint
    evaluation_dir = os.path.join(checkpoint_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)
    return evaluation_dir


def setup_evaluation_subdirectories(evaluation_dir: str) -> dict:
    """Setup evaluation subdirectories and return paths."""
    subdirs = {
        'samples': os.path.join(evaluation_dir, 'generated_samples'),
        'comparisons': os.path.join(evaluation_dir, 'comparisons'),
        'interpolations': os.path.join(evaluation_dir, 'interpolations'),
        'translations': os.path.join(evaluation_dir, 'translations'),
        'side_by_side': os.path.join(evaluation_dir, 'side_by_side'),
        'cycle_consistency': os.path.join(evaluation_dir, 'cycle_consistency')
    }
    
    for subdir_path in subdirs.values():
        os.makedirs(subdir_path, exist_ok=True)
    
    return subdirs


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> Tuple[nn.Module, Dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['args']
    
    if model_args['model'] == 'dcgan':
        model = DCGAN(
            nz=model_args.get('nz', 100),
            ngf=model_args.get('ngf', 64),
            ndf=model_args.get('ndf', 64),
            nc=3,
            image_size=model_args.get('image_size', 64),
            device=device
        )
    elif model_args['model'] == 'cyclegan':
        model = CycleGAN(
            input_nc=3,
            output_nc=3,
            ngf=model_args.get('ngf', 64),
            ndf=model_args.get('ndf', 64),
            n_residual_blocks=model_args.get('n_residual_blocks', 9),
            lambda_cycle=model_args.get('lambda_cycle', 10.0),
            lambda_identity=model_args.get('lambda_identity', 0.5),
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_args['model']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, model_args


def evaluate_dcgan(model: DCGAN, test_loader, args, evaluation_dirs: dict):
    """Evaluate DCGAN model."""
    logger.info("Evaluating DCGAN model...")
    
    # Generate samples
    if args.generate_samples or args.all:
        logger.info("Generating samples...")
        samples = model.generate_samples(args.num_samples)
        
        # Save grid of generated samples
        save_image_grid(
            denormalize_tensor(samples),
            os.path.join(evaluation_dirs['samples'], 'dcgan_generated_samples.png'),
            nrow=8, padding=2
        )
        
        # Save individual samples
        individual_dir = os.path.join(evaluation_dirs['samples'], 'individual')
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            save_image(
                denormalize_tensor(sample),
                os.path.join(individual_dir, f'sample_{i:03d}.png')
            )
        
        logger.info(f"Generated {args.num_samples} samples")
    
    # Latent interpolation
    if args.latent_interpolation or args.all:
        logger.info("Performing latent interpolation...")
        
        # Generate random start and end points
        start_noise = model.generate_noise(1)
        end_noise = model.generate_noise(1)
        
        visualize_latent_interpolation(
            model, start_noise, end_noise,
            os.path.join(evaluation_dirs['interpolations'], 'latent_interpolation.png'),
            num_steps=8
        )
        
        # Multiple interpolations
        for i in range(5):
            start_noise = model.generate_noise(1)
            end_noise = model.generate_noise(1)
            visualize_latent_interpolation(
                model, start_noise, end_noise,
                os.path.join(evaluation_dirs['interpolations'], f'latent_interpolation_{i}.png'),
                num_steps=8
            )
        
        logger.info("Latent interpolation completed")
    
    # Side-by-side comparison with real images
    if args.side_by_side or args.all:
        logger.info("Creating side-by-side comparisons...")
        
        # Get some real images and move to same device as model
        real_batch = next(iter(test_loader))
        real_images = real_batch['black'][:16].to(model.device)  # Move to model device
        
        # Generate same number of fake images
        fake_images = model.generate_samples(16)
        
        # Create comparison
        create_comparison_grid(
            real_images, fake_images,
            os.path.join(evaluation_dirs['comparisons'], 'real_vs_fake_comparison.png')
        )
        
        # Individual side-by-side comparisons
        for i in range(min(8, len(real_images))):
            create_side_by_side_comparison(
                real_images[i], fake_images[i],
                os.path.join(evaluation_dirs['side_by_side'], f'comparison_{i}.png'),
                "Real Image", "Generated Image"
            )
        
        logger.info("Side-by-side comparisons created")


def evaluate_cyclegan(model: CycleGAN, test_loader, args, evaluation_dirs: dict):
    """Evaluate CycleGAN model."""
    logger.info("Evaluating CycleGAN model...")
    
    # Test translations
    if args.test_translations or args.all:
        logger.info("Testing translations...")
        
        # Process multiple batches
        all_results = []
        
        for i, batch in enumerate(test_loader):
            if i >= args.num_samples // args.batch_size:
                break
                
            real_A = batch['black'].to(model.device)  # Move to model device
            real_B = batch['blond'].to(model.device)  # Move to model device
            
            # Limit batch size for visualization
            batch_size = min(4, real_A.size(0))
            real_A = real_A[:batch_size]
            real_B = real_B[:batch_size]
            
            # Generate translations
            translations = model.generate_translations(real_A, real_B)
            all_results.append(translations)
            
            # Save this batch's results
            save_cyclegan_results(
                translations,
                os.path.join(evaluation_dirs['translations'], f'translations_batch_{i}.png'),
                num_samples=batch_size
            )
        
        logger.info(f"Translation testing completed for {len(all_results)} batches")
    
    # Side-by-side comparisons
    if args.side_by_side or args.all:
        logger.info("Creating side-by-side comparisons...")
        
        batch = next(iter(test_loader))
        real_A = batch['black'][:8].to(model.device)  # Black hair - move to model device
        real_B = batch['blond'][:8].to(model.device)  # Blond hair - move to model device
        
        translations = model.generate_translations(real_A, real_B)
        
        # Black to Blond translations
        black_to_blond_dir = os.path.join(evaluation_dirs['side_by_side'], 'black_to_blond')
        os.makedirs(black_to_blond_dir, exist_ok=True)
        
        for i in range(len(real_A)):
            create_side_by_side_comparison(
                real_A[i], translations['fake_B'][i],
                os.path.join(black_to_blond_dir, f'black_to_blond_{i}.png'),
                "Original (Black Hair)", "Translated (Blond Hair)"
            )
        
        # Blond to Black translations
        blond_to_black_dir = os.path.join(evaluation_dirs['side_by_side'], 'blond_to_black')
        os.makedirs(blond_to_black_dir, exist_ok=True)
        
        for i in range(len(real_B)):
            create_side_by_side_comparison(
                real_B[i], translations['fake_A'][i],
                os.path.join(blond_to_black_dir, f'blond_to_black_{i}.png'),
                "Original (Blond Hair)", "Translated (Black Hair)"
            )
        
        # Cycle consistency visualization
        for i in range(len(real_A)):
            # A -> B -> A cycle
            create_comparison_grid(
                torch.stack([real_A[i], translations['fake_B'][i], translations['recovered_A'][i]]),
                torch.zeros_like(torch.stack([real_A[i], translations['fake_B'][i], translations['recovered_A'][i]])),
                os.path.join(evaluation_dirs['cycle_consistency'], f'cycle_A_B_A_{i}.png')
            )
        
        for i in range(len(real_B)):
            # B -> A -> B cycle
            create_comparison_grid(
                torch.stack([real_B[i], translations['fake_A'][i], translations['recovered_B'][i]]),
                torch.zeros_like(torch.stack([real_B[i], translations['fake_A'][i], translations['recovered_B'][i]])),
                os.path.join(evaluation_dirs['cycle_consistency'], f'cycle_B_A_B_{i}.png')
            )
        
        logger.info("Side-by-side comparisons created")


def calculate_fid_score(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    Calculate Frechet Inception Distance (FID) score.
    This is a simplified version - in practice, you'd use a proper implementation.
    """
    # This is a placeholder - implementing full FID requires the Inception network
    # For now, return a dummy value
    logger.warning("FID calculation not implemented - returning dummy value")
    return 0.0


def generate_evaluation_report(model_type: str, results: Dict, evaluation_dir: str):
    """Generate a comprehensive evaluation report."""
    report_path = os.path.join(evaluation_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"=== {model_type.upper()} Evaluation Report ===\n\n")
        f.write(f"Generated samples: {results.get('num_samples', 'N/A')}\n")
        f.write(f"Evaluation time: {results.get('eval_time', 'N/A'):.2f} seconds\n")
        
        if 'fid_score' in results:
            f.write(f"FID Score: {results['fid_score']:.4f}\n")
        
        f.write("\nGenerated Files:\n")
        for file_path in results.get('generated_files', []):
            f.write(f"  - {file_path}\n")
        
        f.write(f"\nAll results saved in: {evaluation_dir}\n")
    
    logger.info(f"Evaluation report saved: {report_path}")


def load_image_for_prediction(image_path: str, image_size: int = 64) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess a single image for prediction, returning tensor and original size."""
    pil_image = Image.open(image_path).convert('RGB')
    original_size = pil_image.size  # (width, height)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    tensor = transform(pil_image)
    # Ensure tensor is a torch.Tensor (it should be after ToTensor())
    assert isinstance(tensor, torch.Tensor), f"Expected tensor, got {type(tensor)}"
    return tensor.unsqueeze(0), original_size  # Add batch dimension


def load_image_for_prediction_optimized(image_path: str, image_size: int = 64, 
                                       training_resolution: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess image with optimal sizing for better quality."""
    pil_image = Image.open(image_path).convert('RGB')
    original_size = pil_image.size  # (width, height)
    
    # If training resolution is provided, resize to that first preserving aspect ratio
    if training_resolution:
        pil_image = resize_crop_center(pil_image, training_resolution)
        logger.info(f"Resized input from {original_size} to {training_resolution} (center crop, maintains aspect ratio)")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    tensor = transform(pil_image)
    assert isinstance(tensor, torch.Tensor), f"Expected tensor, got {type(tensor)}"
    return tensor.unsqueeze(0), original_size  # Add batch dimension


def save_translated_image(tensor: torch.Tensor, output_path: str, target_size: Tuple[int, int]):
    """Save translated tensor as image, resizing to target size."""
    # Denormalize tensor from [-1, 1] to [0, 1]
    denormalized = denormalize_tensor(tensor)
    
    # Convert to PIL image
    if denormalized.dim() == 3:  # Remove batch dimension if present
        img_tensor = denormalized
    else:
        img_tensor = denormalized.squeeze(0)
    
    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(img_tensor.cpu())
    
    # Resize to target size (width, height)
    resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Save
    resized_img.save(output_path)


def predict_cyclegan_on_path(model: CycleGAN, input_path: str, output_dir: str, 
                            direction: str = 'both', image_size: int = 64):
    """Predict CycleGAN translations on single image or folder."""
    logger.info(f"Starting CycleGAN prediction on: {input_path}")
    
    # Determine if input is file or directory
    if os.path.isfile(input_path):
        image_paths = [input_path]
        logger.info("Processing single image")
    elif os.path.isdir(input_path):
        # Get all image files in directory
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend([
                os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(ext.lower())
            ])
        logger.info(f"Found {len(image_paths)} images in directory")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not image_paths:
        raise ValueError("No valid image files found")
    
    # Create output directories
    if direction in ['AtoB', 'both']:
        os.makedirs(os.path.join(output_dir, 'black_to_blond'), exist_ok=True)
    if direction in ['BtoA', 'both']:
        os.makedirs(os.path.join(output_dir, 'blond_to_black'), exist_ok=True)
    if direction == 'both':
        os.makedirs(os.path.join(output_dir, 'side_by_side'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    model.eval()
    
    # Log resolution information
    logger.info(f"Model processes images at {image_size}x{image_size} resolution")
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load and preprocess image, preserving original size
            input_tensor, original_size = load_image_for_prediction(image_path, image_size)
            input_tensor = input_tensor.to(model.device)
            
            # Log resolution info for user awareness
            if i == 0:  # Log once
                logger.info(f"Input image size: {original_size}")
                logger.info(f"Processing at: {image_size}x{image_size}")
                logger.info(f"Output will be upsampled to: {original_size}")
                if original_size[0] > 200 or original_size[1] > 200:
                    logger.warning("Large input image detected - output may be blurry due to resolution downsampling")
                    logger.warning("Consider resizing input to ~178x218 (training resolution) for better quality")
            
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            with torch.no_grad():
                fake_B = None
                fake_A = None
                
                if direction in ['AtoB', 'both']:
                    # Treat input as black hair (domain A) -> translate to blond (domain B)
                    fake_B = model.G_AB(input_tensor)
                    
                    # Save translated result at original size
                    save_translated_image(
                        fake_B[0], 
                        os.path.join(output_dir, 'black_to_blond', f'{filename}_translated.png'),
                        original_size
                    )
                
                if direction in ['BtoA', 'both']:
                    # Treat input as blond hair (domain B) -> translate to black (domain A)
                    fake_A = model.G_BA(input_tensor)
                    
                    # Save translated result at original size
                    save_translated_image(
                        fake_A[0],
                        os.path.join(output_dir, 'blond_to_black', f'{filename}_translated.png'),
                        original_size
                    )
                
                # Create comparisons when both directions are available
                if direction == 'both' and fake_B is not None and fake_A is not None:
                    # Load original image
                    original_img = Image.open(image_path).convert('RGB')
                    
                    # Create translated images at original size
                    fake_B_pil = transforms.ToPILImage()(denormalize_tensor(fake_B[0]).cpu())
                    fake_A_pil = transforms.ToPILImage()(denormalize_tensor(fake_A[0]).cpu())
                    
                    fake_B_resized = fake_B_pil.resize(original_size, Image.Resampling.LANCZOS)
                    fake_A_resized = fake_A_pil.resize(original_size, Image.Resampling.LANCZOS)
                    
                    # Create side-by-side comparison: Original | Black->Blond | Blond->Black
                    comparison_width = original_size[0] * 3
                    comparison_height = original_size[1]
                    comparison_img = Image.new('RGB', (comparison_width, comparison_height))
                    
                    comparison_img.paste(original_img, (0, 0))
                    comparison_img.paste(fake_B_resized, (original_size[0], 0))
                    comparison_img.paste(fake_A_resized, (original_size[0] * 2, 0))
                    
                    # Save in both side_by_side and comparisons folders
                    comparison_img.save(os.path.join(output_dir, 'side_by_side', f'{filename}_comparison.png'))
                    comparison_img.save(os.path.join(output_dir, 'comparisons', f'{filename}_comparison.png'))
                    
                    # Also create a labeled version for comparisons folder
                    # Add text labels for clarity
                    from PIL import ImageDraw, ImageFont
                    labeled_comparison = comparison_img.copy()
                    draw = ImageDraw.Draw(labeled_comparison)
                    
                    try:
                        # Try to use a decent font, fallback to default if not available
                        font_size = max(20, original_size[1] // 30)  # Scale font with image
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # Add labels
                    label_y = 10
                    draw.text((10, label_y), "Original", fill="white", font=font, stroke_width=2, stroke_fill="black")
                    draw.text((original_size[0] + 10, label_y), "→ Blond Hair", fill="white", font=font, stroke_width=2, stroke_fill="black")
                    draw.text((original_size[0] * 2 + 10, label_y), "→ Black Hair", fill="white", font=font, stroke_width=2, stroke_fill="black")
                    
                    labeled_comparison.save(os.path.join(output_dir, 'comparisons', f'{filename}_labeled_comparison.png'))
            
            logger.info(f"Processed {i+1}/{len(image_paths)}: {filename} (original size: {original_size})")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info(f"Prediction completed! Results saved in: {output_dir}")
    logger.info(f"Output images maintain original input dimensions")
    if direction == 'both':
        logger.info(f"Comparison images saved in: {os.path.join(output_dir, 'comparisons')}")


def predict_cyclegan_optimized(model: CycleGAN, input_path: str, output_dir: str, 
                              direction: str = 'both', image_size: int = 64,
                              training_resolution: Tuple[int, int] = (178, 218)):
    """Optimized CycleGAN prediction with better resolution handling."""
    logger.info(f"Starting optimized CycleGAN prediction on: {input_path}")
    logger.info(f"Using training resolution: {training_resolution} for better quality")
    
    # Determine if input is file or directory
    if os.path.isfile(input_path):
        image_paths = [input_path]
        logger.info("Processing single image")
    elif os.path.isdir(input_path):
        # Get all image files in directory
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend([
                os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(ext.lower())
            ])
        logger.info(f"Found {len(image_paths)} images in directory")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not image_paths:
        raise ValueError("No valid image files found")
    
    # Create output directories
    if direction in ['AtoB', 'both']:
        os.makedirs(os.path.join(output_dir, 'black_to_blond'), exist_ok=True)
    if direction in ['BtoA', 'both']:
        os.makedirs(os.path.join(output_dir, 'blond_to_black'), exist_ok=True)
    if direction == 'both':
        os.makedirs(os.path.join(output_dir, 'side_by_side'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    model.eval()
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load and preprocess image with optimized resolution handling
            input_tensor, original_size = load_image_for_prediction_optimized(
                image_path, image_size, training_resolution
            )
            input_tensor = input_tensor.to(model.device)
            
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            with torch.no_grad():
                fake_B = None
                fake_A = None
                
                if direction in ['AtoB', 'both']:
                    fake_B = model.G_AB(input_tensor)
                    save_translated_image(
                        fake_B[0], 
                        os.path.join(output_dir, 'black_to_blond', f'{filename}_translated.png'),
                        training_resolution  # Use training resolution for output
                    )
                
                if direction in ['BtoA', 'both']:
                    fake_A = model.G_BA(input_tensor)
                    save_translated_image(
                        fake_A[0],
                        os.path.join(output_dir, 'blond_to_black', f'{filename}_translated.png'),
                        training_resolution  # Use training resolution for output
                    )
                
                # Create comparisons at training resolution
                if direction == 'both' and fake_B is not None and fake_A is not None:
                    original_img = Image.open(image_path).convert('RGB').resize(training_resolution, Image.Resampling.LANCZOS)
                    
                    fake_B_pil = transforms.ToPILImage()(denormalize_tensor(fake_B[0]).cpu())
                    fake_A_pil = transforms.ToPILImage()(denormalize_tensor(fake_A[0]).cpu())
                    
                    fake_B_resized = fake_B_pil.resize(training_resolution, Image.Resampling.LANCZOS)
                    fake_A_resized = fake_A_pil.resize(training_resolution, Image.Resampling.LANCZOS)
                    
                    # Create side-by-side comparison at training resolution
                    comparison_width = training_resolution[0] * 3
                    comparison_height = training_resolution[1]
                    comparison_img = Image.new('RGB', (comparison_width, comparison_height))
                    
                    comparison_img.paste(original_img, (0, 0))
                    comparison_img.paste(fake_B_resized, (training_resolution[0], 0))
                    comparison_img.paste(fake_A_resized, (training_resolution[0] * 2, 0))
                    
                    comparison_img.save(os.path.join(output_dir, 'side_by_side', f'{filename}_comparison.png'))
                    comparison_img.save(os.path.join(output_dir, 'comparisons', f'{filename}_comparison.png'))
            
            logger.info(f"Processed {i+1}/{len(image_paths)}: {filename} (output size: {training_resolution})")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info(f"Optimized prediction completed! Results saved in: {output_dir}")
    logger.info(f"Output images saved at training resolution: {training_resolution}")


def resize_maintain_aspect_ratio(image: Image.Image, target_size: Tuple[int, int], 
                                 fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Resize image to target size while maintaining aspect ratio.
    Pads with fill_color if needed.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate scaling factor to fit within target size
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
    
    # Calculate new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and fill color
    result_image = Image.new('RGB', target_size, fill_color)
    
    # Paste resized image in center
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    result_image.paste(resized_image, (paste_x, paste_y))
    
    return result_image


def resize_crop_center(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image to target size by scaling and center cropping.
    This maintains aspect ratio but crops parts of the image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate scaling factor to cover target size
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = max(scale_w, scale_h)  # Use larger scale to cover entire target
    
    # Calculate new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate crop box for center crop
    crop_x = (new_width - target_width) // 2
    crop_y = (new_height - target_height) // 2
    crop_box = (crop_x, crop_y, crop_x + target_width, crop_y + target_height)
    
    # Crop to target size
    result_image = resized_image.crop(crop_box)
    
    return result_image


def resize_stretch(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image to target size by stretching (current method).
    This does NOT maintain aspect ratio.
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def visualize_resize_methods(image_path: str, target_size: Tuple[int, int] = (178, 218)):
    """
    Visualize different resizing methods for comparison.
    Perfect for use in Jupyter notebooks!
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Apply different resizing methods
    method1 = resize_maintain_aspect_ratio(original_img, target_size)
    method2 = resize_crop_center(original_img, target_size)
    method3 = resize_stretch(original_img, target_size)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original\n{original_img.size}')
    axes[0, 0].axis('off')
    
    # Method 1: Maintain aspect ratio with padding
    axes[0, 1].imshow(method1)
    axes[0, 1].set_title(f'Maintain Aspect Ratio\n{method1.size}\n(with padding)')
    axes[0, 1].axis('off')
    
    # Method 2: Center crop
    axes[1, 0].imshow(method2)
    axes[1, 0].set_title(f'Center Crop\n{method2.size}\n(maintains ratio, crops)')
    axes[1, 0].axis('off')
    
    # Method 3: Stretch (current method)
    axes[1, 1].imshow(method3)
    axes[1, 1].set_title(f'Stretch (Current)\n{method3.size}\n(distorts image)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print aspect ratios
    orig_ratio = original_img.size[0] / original_img.size[1]
    target_ratio = target_size[0] / target_size[1]
    
    print(f"Original aspect ratio: {orig_ratio:.3f}")
    print(f"Target aspect ratio: {target_ratio:.3f}")
    print(f"Aspect ratio difference: {abs(orig_ratio - target_ratio):.3f}")
    
    return {
        'original': original_img,
        'maintain_aspect': method1,
        'center_crop': method2,
        'stretch': method3
    }


def load_image_like_dataloader(image_path: str, image_size: int = 64) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess image exactly like the dataloader does (no double resizing)."""
    pil_image = Image.open(image_path).convert('RGB')
    original_size = pil_image.size  # (width, height)
    
    # Use EXACT same transforms as dataloader (no double resizing!)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Direct resize like dataloader
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Same normalization
    ])
    
    tensor = transform(pil_image)
    assert isinstance(tensor, torch.Tensor), f"Expected tensor, got {type(tensor)}"
    logger.info(f"Loaded image using dataloader transforms: {original_size} → {image_size}×{image_size}")
    return tensor.unsqueeze(0), original_size  # Add batch dimension


def predict_cyclegan_like_batch(model: CycleGAN, input_path: str, output_dir: str, 
                               direction: str = 'both', image_size: int = 64):
    """Predict CycleGAN translations using the exact same process as translation batches."""
    logger.info(f"Starting batch-like CycleGAN prediction on: {input_path}")
    logger.info(f"Using EXACT same process as translation batch images")
    
    # Determine if input is file or directory
    if os.path.isfile(input_path):
        image_paths = [input_path]
        logger.info("Processing single image")
    elif os.path.isdir(input_path):
        # Get all image files in directory
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend([
                os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(ext.lower())
            ])
        logger.info(f"Found {len(image_paths)} images in directory")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not image_paths:
        raise ValueError("No valid image files found")
    
    # Create output directories
    if direction in ['AtoB', 'both']:
        os.makedirs(os.path.join(output_dir, 'black_to_blond'), exist_ok=True)
    if direction in ['BtoA', 'both']:
        os.makedirs(os.path.join(output_dir, 'blond_to_black'), exist_ok=True)
    if direction == 'both':
        os.makedirs(os.path.join(output_dir, 'side_by_side'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    model.eval()
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load image EXACTLY like dataloader (no double resizing!)
            input_tensor, original_size = load_image_like_dataloader(image_path, image_size)
            input_tensor = input_tensor.to(model.device)
            
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            with torch.no_grad():
                fake_B = None
                fake_A = None
                
                if direction in ['AtoB', 'both']:
                    # Create a fake "batch" to simulate translation batch process
                    fake_real_B = torch.zeros_like(input_tensor)  # Dummy B image
                    
                    # Use the same generate_translations function as batch process
                    translations = model.generate_translations(input_tensor, fake_real_B)
                    
                    # Save individual results
                    fake_B = translations['fake_B'][0]  # Black->Blond result
                    
                    # Save using same method as translation batch (native tensor)
                    save_image(
                        denormalize_tensor(fake_B),
                        os.path.join(output_dir, 'black_to_blond', f'{filename}_translated.png')
                    )
                
                if direction in ['BtoA', 'both']:
                    # Create a fake "batch" to simulate translation batch process  
                    fake_real_A = torch.zeros_like(input_tensor)  # Dummy A image
                    
                    # Use the same generate_translations function as batch process
                    translations = model.generate_translations(fake_real_A, input_tensor)
                    
                    # Save individual results
                    fake_A = translations['fake_A'][0]  # Blond->Black result
                    
                    # Save using same method as translation batch (native tensor)
                    save_image(
                        denormalize_tensor(fake_A),
                        os.path.join(output_dir, 'blond_to_black', f'{filename}_translated.png')
                    )
                
                # Create 3-way side-by-side comparison: Original | A2B | B2A
                if direction == 'both' and fake_B is not None and fake_A is not None:
                    original_tensor = input_tensor[0]
                    
                    # Create 3-way comparison using PIL
                    original_pil = tensor_to_pil(original_tensor)
                    fake_B_pil = tensor_to_pil(fake_B)
                    fake_A_pil = tensor_to_pil(fake_A)
                    
                    # Create side-by-side comparison: Original | Black->Blond | Blond->Black
                    width, height = original_pil.size
                    comparison_width = width * 3
                    comparison_height = height
                    comparison_img = Image.new('RGB', (comparison_width, comparison_height))
                    
                    comparison_img.paste(original_pil, (0, 0))
                    comparison_img.paste(fake_B_pil, (width, 0))
                    comparison_img.paste(fake_A_pil, (width * 2, 0))
                    
                    # Save comparison image
                    comparison_img.save(os.path.join(output_dir, 'side_by_side', f'{filename}_comparison.png'))
                    comparison_img.save(os.path.join(output_dir, 'comparisons', f'{filename}_comparison.png'))
                    
                    # Also create a labeled version for comparisons folder
                    from PIL import ImageDraw, ImageFont
                    labeled_comparison = comparison_img.copy()
                    draw = ImageDraw.Draw(labeled_comparison)
                    
                    try:
                        # Try to use a decent font, fallback to default if not available
                        font_size = max(16, height // 20)  # Scale font with image
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # Add labels
                    label_y = 10
                    draw.text((width//2 - 20, label_y), "Original", fill="white", font=font, stroke_width=2, stroke_fill="black")
                    draw.text((width + width//2 - 30, label_y), "→ Blond Hair", fill="white", font=font, stroke_width=2, stroke_fill="black")
                    draw.text((width*2 + width//2 - 30, label_y), "→ Black Hair", fill="white", font=font, stroke_width=2, stroke_fill="black")
                    
                    labeled_comparison.save(os.path.join(output_dir, 'comparisons', f'{filename}_labeled_comparison.png'))
            
            logger.info(f"Processed {i+1}/{len(image_paths)}: {filename} (native {image_size}×{image_size})")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info(f"Batch-like prediction completed! Results saved in: {output_dir}")
    logger.info(f"Images processed at native {image_size}×{image_size} resolution (no upsampling)")
    logger.info(f"Side-by-side comparisons: Original | A→B | B→A in single images")


def main():
    args = get_args()
    
    # Validate arguments
    if args.predict and not args.input_path:
        raise ValueError("--predict requires --input_path to be specified")
    
    if not args.predict and not args.data_root:
        raise ValueError("--data_root is required unless using --predict mode")
    
    # Determine evaluation directory
    evaluation_dir = determine_evaluation_directory(args.checkpoint, args.save_dir)
    
    # Setup logging
    setup_logging(os.path.join(evaluation_dir, 'evaluation.log'))
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = str(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluation directory: {evaluation_dir}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, model_args = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Determine actual model type from checkpoint (not from command line args)
    actual_model_type = model_args['model']
    logger.info(f"Detected model type: {actual_model_type}")
    
    # Handle prediction mode for CycleGAN
    if args.predict:
        if actual_model_type != 'cyclegan':
            raise ValueError("--predict mode is only supported for CycleGAN models")
        
        logger.info(f"Running prediction on: {args.input_path}")
        # Cast model to CycleGAN type for prediction
        assert isinstance(model, CycleGAN), "Model should be CycleGAN for prediction mode"
        
        # Use the actual image size from model training, not the default
        model_image_size = model_args.get('image_size', 64)
        logger.info(f"Using model's training image size: {model_image_size}")
        
        # Adjust training resolution based on model's image size
        if model_image_size == 128:
            training_resolution = (356, 436)  # 2x the original 178x218
        elif model_image_size == 64:
            training_resolution = (178, 218)  # Original size
        else:
            training_resolution = (178, 218)  # Fallback
        
        logger.info(f"Using training resolution: {training_resolution} for image size {model_image_size}")
        
        if args.batch_quality:
            predict_cyclegan_like_batch(
                model, 
                args.input_path, 
                evaluation_dir, 
                args.direction, 
                model_image_size
            )
        else:
            predict_cyclegan_optimized(
                model, 
                args.input_path, 
                evaluation_dir, 
                args.direction, 
                model_image_size,  # Use correct image size
                training_resolution
            )
        logger.info("Prediction completed!")
        return
    
    # Standard evaluation mode (requires data_root)
    evaluation_dirs = setup_evaluation_subdirectories(evaluation_dir)
    
    # Use the actual image size from model training, not command line args
    model_image_size = model_args.get('image_size', 64)
    logger.info(f"Using model's training image size: {model_image_size} (overriding command line args)")
    
    # Create dataloaders with correct image size
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=model_image_size,  # Use model's image size
        num_workers=args.num_workers
    )
    
    # Start evaluation
    logger.info("Starting evaluation...")
    import time
    start_time = time.time()
    
    generated_files = []
    
    # Call appropriate evaluation function based on actual model type
    if actual_model_type == 'dcgan':
        assert isinstance(model, DCGAN), "Model should be DCGAN"
        # Update args to use model's image size
        args.image_size = model_image_size
        evaluate_dcgan(model, test_loader, args, evaluation_dirs)
    elif actual_model_type == 'cyclegan':
        assert isinstance(model, CycleGAN), "Model should be CycleGAN"
        # Update args to use model's image size
        args.image_size = model_image_size
        evaluate_cyclegan(model, test_loader, args, evaluation_dirs)
    else:
        raise ValueError(f"Unknown model type: {actual_model_type}")
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    # Generate report
    results = {
        'num_samples': args.num_samples,
        'eval_time': eval_time,
        'generated_files': generated_files
    }
    
    generate_evaluation_report(actual_model_type, results, evaluation_dir)
    
    logger.info("Evaluation completed!")
    logger.info(f"Results saved in: {evaluation_dir}")


if __name__ == '__main__':
    main() 