# GANs from Scratch: Black-Blond Hair Translation

This repository implements DCGAN and CycleGAN models for black-to-blond hair translation using PyTorch. It includes comprehensive training, evaluation, and visualization tools with TensorBoard logging and loguru for structured logging.

## Features

- **DCGAN Implementation**: Deep Convolutional GAN for generating hair images
- **CycleGAN Implementation**: Bidirectional hair color translation (black ↔ blond)
- **Black-Blond Dataset**: Custom dataset class with train/validation/test splits
- **Comprehensive Training**: Full training pipeline with checkpointing and resuming
- **Evaluation Tools**: Model evaluation with side-by-side comparisons and metrics
- **Visualization**: TensorBoard integration and custom visualization utilities
- **Logging**: Structured logging with loguru for better debugging
- **Experiment Management**: Organized experiment structure with automatic naming

## Directory Structure

```
src/
├── datasets/
│   ├── __init__.py
│   └── black_blond_dataset.py     # Dataset implementation
├── models/
│   ├── __init__.py
│   ├── dcgan.py                   # DCGAN implementation
│   └── cyclegan.py                # CycleGAN implementation
├── utils/
│   ├── __init__.py
│   ├── visualization.py           # Visualization utilities
│   └── logging_utils.py           # Logging utilities
├── train.py                       # Main training script
└── evaluate.py                    # Evaluation script

experiments/                       # Auto-generated experiments
├── data_dcgan_16_100/            # dataset_model_batch_epochs
│   ├── checkpoints/
│   ├── samples/
│   ├── tensorboard/
│   ├── config.json
│   ├── training.log
│   └── evaluation/               # (after running evaluation)
└── data_cyclegan_8_200/
    └── ...
```

## Experiment Organization

All experiments are automatically organized in the `experiments/` directory using the naming convention:
**`{dataset_name}_{model}_{batch_size}_{epochs}`**

For example:
- `celeba_dcgan_16_100` - DCGAN trained on CelebA data with batch size 16 for 100 epochs
- `custom_cyclegan_8_200` - CycleGAN trained on custom data with batch size 8 for 200 epochs

Each experiment contains:
- `checkpoints/` - Model checkpoints
- `samples/` - Generated samples during training  
- `tensorboard/` - TensorBoard logs
- `config.json` - Training configuration
- `training.log` - Detailed training logs
- `evaluation/` - Evaluation results (created when evaluation is run)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gans-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

Organize your data in the following structure:
```
data/
├── black/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── blond/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The dataset class automatically splits the data into train (70%), validation (15%), and test (15%) sets.

## Training

### DCGAN Training

Train a DCGAN model to generate hair images:

```bash
python src/train.py \
    --model dcgan \
    --data_root ./data \
    --epochs 100 \
    --batch_size 16 \
    --image_size 64
```

This creates: `experiments/data_dcgan_16_100/`

### CycleGAN Training

Train a CycleGAN model for bidirectional hair color translation:

```bash
python src/train.py \
    --model cyclegan \
    --data_root ./data \
    --epochs 200 \
    --batch_size 8 \
    --image_size 64 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5
```

This creates: `experiments/data_cyclegan_8_200/`

### Training Parameters

- `--model`: Choose between `dcgan` or `cyclegan`
- `--data_root`: Path to directory containing black and blond folders
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--image_size`: Size to resize images to (default: 64)
- `--lr`: Learning rate (default: 0.0002)
- `--save_freq`: Save checkpoint every N epochs (default: 10)
- `--log_freq`: Log training stats every N iterations (default: 100)
- `--sample_freq`: Generate samples every N iterations (default: 500)
- `--resume`: Path to checkpoint to resume training from

### CycleGAN Specific Parameters

- `--lambda_cycle`: Weight for cycle consistency loss (default: 10.0)
- `--lambda_identity`: Weight for identity loss (default: 0.5)
- `--n_residual_blocks`: Number of residual blocks in generator (default: 9)

## Evaluation

Evaluation automatically detects the experiment structure and saves results in the same experiment directory.

### DCGAN Evaluation

```bash
python src/evaluate.py \
    --model dcgan \
    --checkpoint experiments/data_dcgan_16_100/checkpoints/dcgan_epoch_99.pth \
    --data_root ./data \
    --all
```

Results saved in: `experiments/data_dcgan_16_100/evaluation/`

### CycleGAN Evaluation

```bash
python src/evaluate.py \
    --model cyclegan \
    --checkpoint experiments/data_cyclegan_8_200/checkpoints/cyclegan_epoch_199.pth \
    --data_root ./data \
    --all
```

Results saved in: `experiments/data_cyclegan_8_200/evaluation/`

### Evaluation Options

- `--generate_samples`: Generate samples from the model
- `--test_translations`: Test translations (CycleGAN only)
- `--latent_interpolation`: Perform latent space interpolation (DCGAN only)
- `--side_by_side`: Create side-by-side comparisons
- `--all`: Run all evaluation modes

### Evaluation Structure

The evaluation creates organized subdirectories:
```
evaluation/
├── generated_samples/
├── comparisons/
├── side_by_side/
├── interpolations/          # DCGAN only
├── translations/            # CycleGAN only
├── cycle_consistency/       # CycleGAN only
├── evaluation.log
└── evaluation_report.txt
```

## Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir experiments/data_cyclegan_8_200/tensorboard
```

### Log Files

Training logs are saved in each experiment directory:
- `training.log`: Detailed training logs with loguru
- `config.json`: Training configuration
- `tensorboard/`: TensorBoard event files

## Model Architecture

### DCGAN

- **Generator**: 5-layer transposed convolution network with batch normalization
- **Discriminator**: 5-layer convolution network with batch normalization
- **Loss**: Binary cross-entropy loss
- **Optimization**: Adam optimizer with β₁=0.5, β₂=0.999

### CycleGAN

- **Generators**: ResNet-based architecture with 9 residual blocks
- **Discriminators**: PatchGAN discriminators
- **Losses**: 
  - Adversarial loss (MSE)
  - Cycle consistency loss (L1)
  - Identity loss (L1)
- **Optimization**: Adam optimizer with β₁=0.5, β₂=0.999

## Results

The evaluation script generates:

### DCGAN Results
- Generated sample grids
- Individual sample images
- Latent space interpolations
- Real vs. fake comparisons

### CycleGAN Results
- Black-to-blond translations
- Blond-to-black translations
- Cycle consistency visualizations
- Side-by-side comparisons

## Example Usage

### Quick Start: Train CycleGAN

```bash
# Assuming your data is in ./data/
python src/train.py \
    --model cyclegan \
    --data_root ./data \
    --epochs 50 \
    --batch_size 4
```

This creates: `experiments/data_cyclegan_4_50/`

### Monitor Training

```bash
tensorboard --logdir experiments/data_cyclegan_4_50/tensorboard
```

### Evaluate and Generate Comparisons

```bash
python src/evaluate.py \
    --model cyclegan \
    --checkpoint experiments/data_cyclegan_4_50/checkpoints/cyclegan_epoch_49.pth \
    --data_root ./data \
    --all
```

Results in: `experiments/data_cyclegan_4_50/evaluation/`

## Working with Multiple Experiments

The organized structure makes it easy to compare experiments:

```bash
experiments/
├── celeba_dcgan_16_100/     # High-quality DCGAN
├── celeba_dcgan_32_100/     # Large batch DCGAN
├── celeba_cyclegan_8_200/   # Standard CycleGAN
└── celeba_cyclegan_4_300/   # Long training CycleGAN
```

Compare TensorBoard logs:
```bash
tensorboard --logdir experiments/ --port 6006
```

## Tips for Best Results

1. **Data Quality**: Ensure your black and blond hair images are well-cropped and aligned
2. **Training Time**: CycleGAN typically requires 100-200 epochs for good results
3. **Batch Size**: Use smaller batch sizes if you encounter memory issues
4. **Monitoring**: Watch the TensorBoard logs to ensure stable training
5. **Checkpointing**: Checkpoints are automatically saved every 10 epochs
6. **Experiment Naming**: The automatic naming helps track different configurations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Training Instability**: Try different learning rates or lambda values
3. **Poor Quality Results**: Increase training epochs or check data quality

### Debug Mode

Add `--log_freq 10` to get more frequent logging during training.

### Finding Experiments

List all experiments:
```bash
ls experiments/
```

Find specific model experiments:
```bash
ls experiments/*dcgan*
ls experiments/*cyclegan*
```

## Contributing

Feel free to open issues or submit pull requests to improve the implementation.

## License

This project is open source. See the repository for license details.