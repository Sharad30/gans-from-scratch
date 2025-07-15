"""
Resize functions for image preprocessing in CycleGAN evaluation.
Copy these functions to your Jupyter notebook to visualize resizing methods.
"""

from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def resize_maintain_aspect_ratio(image: Image.Image, target_size: Tuple[int, int], 
                                fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Resize image to target size while maintaining aspect ratio.
    Pads with fill_color if needed.
    
    Args:
        image: PIL Image to resize
        target_size: (width, height) tuple
        fill_color: RGB tuple for padding color
    
    Returns:
        Resized PIL Image
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
    
    Args:
        image: PIL Image to resize
        target_size: (width, height) tuple
    
    Returns:
        Resized PIL Image
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
    Resize image to target size by stretching (distorts aspect ratio).
    This does NOT maintain aspect ratio.
    
    Args:
        image: PIL Image to resize
        target_size: (width, height) tuple
    
    Returns:
        Resized PIL Image
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def visualize_resize_methods(image_path: str, target_size: Tuple[int, int] = (178, 218)) -> Dict:
    """
    Visualize different resizing methods for comparison.
    Perfect for use in Jupyter notebooks!
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height)
    
    Returns:
        Dictionary containing all resized images
    """
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


def analyze_your_image(image_path: str):
    """
    Analyze your specific image and training data compatibility.
    
    Args:
        image_path: Path to your image
    """
    img = Image.open(image_path).convert('RGB')
    
    print("=== IMAGE ANALYSIS ===")
    print(f"Your image size: {img.size}")
    print(f"Your aspect ratio: {img.size[0] / img.size[1]:.3f}")
    
    print("\n=== TRAINING DATA INFO ===")
    print(f"Training resolution: (178, 218)")
    print(f"Training aspect ratio: {178 / 218:.3f}")
    
    print("\n=== PROCESSING PIPELINE ===")
    print("1. Your image → Training resolution (178×218)")
    print("2. Training resolution → Model processing (64×64)")
    print("3. Model output (64×64) → Training resolution (178×218)")
    
    print("\n=== RECOMMENDATIONS ===")
    your_ratio = img.size[0] / img.size[1]
    training_ratio = 178 / 218
    
    if abs(your_ratio - training_ratio) < 0.1:
        print("✅ Your image aspect ratio is close to training data - good results expected")
    else:
        print("⚠️  Your image aspect ratio differs from training data")
        print("   Center crop will be used to maintain aspect ratio")
        print("   Some parts of your image will be cropped")
    
    if max(img.size) > 400:
        print("⚠️  Large input image detected - significant downsampling will occur")
    
    # Show what will happen with center crop
    cropped = resize_crop_center(img, (178, 218))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Original\n{img.size}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cropped)
    plt.title(f'After Center Crop\n{cropped.size}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_model_quality_issues(checkpoint_path: str, image_path: str):
    """
    Analyze potential quality issues with CycleGAN model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        image_path: Path to your test image
    """
    import torch
    import json
    import os
    
    print("=== MODEL QUALITY ANALYSIS ===")
    
    # Load checkpoint and config
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_args = checkpoint['args']
        
        # Try to find config.json
        config_path = None
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir.endswith('checkpoints'):
            experiment_dir = os.path.dirname(checkpoint_dir)
            config_path = os.path.join(experiment_dir, 'config.json')
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Found config: {config_path}")
        else:
            config = model_args
            print("⚠️  Using args from checkpoint (config.json not found)")
        
        # Analysis
        print(f"\n=== TRAINING CONFIGURATION ===")
        print(f"Model: {config.get('model', 'Unknown')}")
        print(f"Epochs trained: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Training image size: {config.get('image_size', 'Unknown')}")
        print(f"Batch size: {config.get('batch_size', 'Unknown')}")
        print(f"Learning rate: {config.get('lr', 'Unknown')}")
        
        # Quality assessment
        print(f"\n=== QUALITY ASSESSMENT ===")
        
        epochs = checkpoint.get('epoch', 0)
        image_size = config.get('image_size', 64)
        
        quality_issues = []
        
        # Check epochs
        if epochs < 50:
            quality_issues.append(f"❌ CRITICAL: Only {epochs} epochs trained (need 100-200 minimum)")
        elif epochs < 100:
            quality_issues.append(f"⚠️  WARNING: Only {epochs} epochs trained (recommend 100-200)")
        else:
            print(f"✅ Training epochs: {epochs} (good)")
        
        # Check image size
        if image_size < 64:
            quality_issues.append(f"❌ CRITICAL: Very low training resolution ({image_size}x{image_size})")
        elif image_size == 64:
            quality_issues.append(f"⚠️  WARNING: Low training resolution ({image_size}x{image_size})")
        else:
            print(f"✅ Training resolution: {image_size}x{image_size} (good)")
        
        # Check your input image
        print(f"\n=== YOUR INPUT IMAGE ===")
        img = Image.open(image_path).convert('RGB')
        print(f"Your image size: {img.size}")
        print(f"Your aspect ratio: {img.size[0] / img.size[1]:.3f}")
        
        # Resolution comparison
        max_input_size = max(img.size)
        resolution_ratio = max_input_size / image_size
        
        if resolution_ratio > 5:
            quality_issues.append(f"❌ MAJOR: Input image much larger than training ({max_input_size} vs {image_size})")
        elif resolution_ratio > 2:
            quality_issues.append(f"⚠️  WARNING: Input image larger than training ({max_input_size} vs {image_size})")
        
        # Print issues
        if quality_issues:
            print(f"\n=== QUALITY ISSUES FOUND ===")
            for issue in quality_issues:
                print(issue)
        else:
            print(f"\n✅ No major quality issues detected")
        
        # Recommendations
        print(f"\n=== RECOMMENDATIONS ===")
        
        if epochs < 100:
            print("1. 🚀 TRAIN LONGER:")
            print("   - Resume training for more epochs")
            print("   - CycleGAN needs 100-200 epochs minimum")
            print("   - Use --resume with your checkpoint")
        
        if image_size <= 64:
            print("2. 🎯 TRAIN AT HIGHER RESOLUTION:")
            print("   - Retrain with --image_size 128 or 256")
            print("   - Higher resolution = better quality")
            print("   - Will take longer but much better results")
        
        if resolution_ratio > 2:
            print("3. 📐 RESIZE YOUR INPUT:")
            print(f"   - Resize your {img.size} image to ~{image_size*2}x{image_size*2}")
            print("   - Or use the updated evaluation script (auto-detects)")
        
        print("\n4. 📊 IMMEDIATE FIXES:")
        print("   - Use the updated evaluation script")
        print("   - It now auto-detects the correct image size")
        print("   - Run the same command again for better results")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        print("Make sure the checkpoint path is correct")


# Example usage for your notebook:
"""
# Copy and run this in your Jupyter notebook:

from PIL import Image
import matplotlib.pyplot as plt

# Replace with your image path
image_path = "path/to/your/image.jpg"

# Analyze your specific image
analyze_your_image(image_path)

# Visualize all resize methods
results = visualize_resize_methods(image_path, target_size=(178, 218))

# Access individual results
original = results['original']
center_crop = results['center_crop']
maintain_aspect = results['maintain_aspect']
stretch = results['stretch']
""" 