import os
import sys
from loguru import logger
from typing import Dict, Any, List
import json
import time


def setup_logging(log_file: str = None, level: str = "INFO") -> None:
    """
    Setup loguru logger with file and console handlers.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="7 days"
        )
    
    logger.info("Logging setup completed")


def log_losses(losses: Dict[str, float], step: int, prefix: str = "") -> None:
    """
    Log training losses in a formatted way.
    
    Args:
        losses: Dictionary of loss names and values
        step: Current training step
        prefix: Optional prefix for logging
    """
    loss_str = " | ".join([f"{name}: {value:.4f}" for name, value in losses.items()])
    logger.info(f"{prefix}Step {step} - {loss_str}")


def log_model_summary(model, input_shape: tuple = None) -> None:
    """
    Log model architecture summary.
    
    Args:
        model: PyTorch model
        input_shape: Input shape for the model
    """
    logger.info("Model Architecture:")
    logger.info(f"Model type: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log model structure
    if hasattr(model, 'generator'):
        gen_params = sum(p.numel() for p in model.generator.parameters())
        logger.info(f"Generator parameters: {gen_params:,}")
    
    if hasattr(model, 'discriminator'):
        disc_params = sum(p.numel() for p in model.discriminator.parameters())
        logger.info(f"Discriminator parameters: {disc_params:,}")
    
    if hasattr(model, 'G_AB'):
        g_ab_params = sum(p.numel() for p in model.G_AB.parameters())
        g_ba_params = sum(p.numel() for p in model.G_BA.parameters())
        d_a_params = sum(p.numel() for p in model.D_A.parameters())
        d_b_params = sum(p.numel() for p in model.D_B.parameters())
        
        logger.info(f"Generator A->B parameters: {g_ab_params:,}")
        logger.info(f"Generator B->A parameters: {g_ba_params:,}")
        logger.info(f"Discriminator A parameters: {d_a_params:,}")
        logger.info(f"Discriminator B parameters: {d_b_params:,}")


def save_training_log(log_data: Dict[str, Any], save_path: str) -> None:
    """
    Save training log data to JSON file.
    
    Args:
        log_data: Dictionary containing training metrics
        save_path: Path to save the log file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"Training log saved to: {save_path}")


class TrainingMetrics:
    """Class to track and manage training metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> float:
        """Get the latest value for a metric."""
        return self.metrics[key][-1] if key in self.metrics and self.metrics[key] else 0.0
    
    def get_average(self, key: str, last_n: int = None) -> float:
        """Get average value for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all metrics."""
        return self.metrics.copy()
    
    def save_to_file(self, filepath: str) -> None:
        """Save metrics to file."""
        data = {
            'metrics': self.metrics,
            'training_time': time.time() - self.start_time,
            'timestamp': time.time()
        }
        save_training_log(data, filepath)
    
    def log_summary(self, epoch: int = None) -> None:
        """Log a summary of current metrics."""
        if epoch is not None:
            logger.info(f"=== Epoch {epoch} Summary ===")
        else:
            logger.info("=== Training Summary ===")
        
        for key, values in self.metrics.items():
            if values:
                latest = values[-1]
                avg = sum(values) / len(values)
                logger.info(f"{key}: Latest={latest:.4f}, Average={avg:.4f}")
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"Training time: {elapsed_time:.2f} seconds")


def log_system_info() -> None:
    """Log system information."""
    import torch
    import platform
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")


def create_progress_logger(total_steps: int, log_interval: int = 100):
    """
    Create a progress logger for training loops.
    
    Args:
        total_steps: Total number of training steps
        log_interval: Interval for logging progress
    
    Returns:
        Progress logger function
    """
    start_time = time.time()
    
    def log_progress(current_step: int, losses: Dict[str, float] = None):
        if current_step % log_interval == 0 or current_step == total_steps:
            elapsed = time.time() - start_time
            progress = current_step / total_steps * 100
            eta = elapsed / current_step * (total_steps - current_step) if current_step > 0 else 0
            
            progress_msg = f"Progress: {progress:.1f}% ({current_step}/{total_steps}) "
            progress_msg += f"Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s"
            
            if losses:
                loss_msg = " | " + " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                progress_msg += loss_msg
            
            logger.info(progress_msg)
    
    return log_progress 