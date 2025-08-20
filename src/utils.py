"""
Utility functions for the image colorization project.
Contains helper functions, visualization tools, and common operations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from skimage import color
from typing import List, Tuple, Optional, Dict, Any
import yaml
import logging
from pathlib import Path
import json
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_project_directories(base_dir: str = ".") -> Dict[str, str]:
    """
    Create project directory structure.
    
    Args:
        base_dir: Base directory for the project
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        'data_raw': os.path.join(base_dir, 'data', 'raw'),
        'data_processed': os.path.join(base_dir, 'data', 'processed'),
        'data_augmented': os.path.join(base_dir, 'data', 'augmented'),
        'models': os.path.join(base_dir, 'models'),
        'models_checkpoints': os.path.join(base_dir, 'models', 'checkpoints'),
        'models_baseline': os.path.join(base_dir, 'models', 'baseline_model'),
        'models_augmented': os.path.join(base_dir, 'models', 'augmented_model'),
        'results': os.path.join(base_dir, 'results'),
        'results_baseline': os.path.join(base_dir, 'results', 'baseline'),
        'results_augmented': os.path.join(base_dir, 'results', 'augmented'),
        'results_comparisons': os.path.join(base_dir, 'results', 'comparisons'),
        'results_logs': os.path.join(base_dir, 'results', 'logs'),
        'config': os.path.join(base_dir, 'config'),
        'src': os.path.join(base_dir, 'src'),
        'notebooks': os.path.join(base_dir, 'notebooks')
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return directories


def create_dataset_splits(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        source_dir: Directory containing all images
        output_dir: Base output directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split counts
    """
    import random
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(source_dir).glob(f"*{ext}"))
        image_files.extend(Path(source_dir).glob(f"*{ext.upper()}"))
    
    # Shuffle files
    image_files = list(image_files)
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_idx = int(total_files * train_ratio)
    val_idx = int(total_files * (train_ratio + val_ratio))
    
    # Split files
    train_files = image_files[:train_idx]
    val_files = image_files[train_idx:val_idx]
    test_files = image_files[val_idx:]
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy files
    for file in train_files:
        shutil.copy2(file, train_dir)
    
    for file in val_files:
        shutil.copy2(file, val_dir)
    
    for file in test_files:
        shutil.copy2(file, test_dir)
    
    split_info = {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'total': total_files
    }
    
    logger.info(f"Dataset split completed: {split_info}")
    
    # Save split information
    split_info_path = os.path.join(output_dir, 'split_info.json')
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return split_info


def visualize_color_distribution(
    image_path: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Visualize color distribution of an image in LAB color space.
    
    Args:
        image_path: Path to the image
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Convert to LAB
    lab_image = color.rgb2lab(image_np)
    
    # Extract channels
    L = lab_image[:, :, 0].flatten()
    A = lab_image[:, :, 1].flatten()
    B = lab_image[:, :, 2].flatten()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(image_np.mean(axis=2), cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # LAB visualization
    axes[0, 2].imshow(lab_image[:, :, 0], cmap='gray')
    axes[0, 2].set_title('L Channel (Lightness)')
    axes[0, 2].axis('off')
    
    # Histograms
    axes[1, 0].hist(L, bins=50, alpha=0.7, color='gray')
    axes[1, 0].set_title('L Channel Distribution')
    axes[1, 0].set_xlabel('Lightness')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(A, bins=50, alpha=0.7, color='red')
    axes[1, 1].set_title('A Channel Distribution')
    axes[1, 1].set_xlabel('Green-Red')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].hist(B, bins=50, alpha=0.7, color='blue')
    axes[1, 2].set_title('B Channel Distribution')
    axes[1, 2].set_xlabel('Blue-Yellow')
    axes[1, 2].set_ylabel('Frequency')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Color distribution plot saved to {save_path}")
    
    plt.show()


def create_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Create and display training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
        title: Title for the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add best validation point
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    plt.scatter(best_val_idx + 1, best_val_loss, color='red', s=100, zorder=5)
    plt.annotate(
        f'Best: {best_val_loss:.4f}',
        xy=(best_val_idx + 1, best_val_loss),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    plt.show()


def compare_images_side_by_side(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Display multiple images side by side for comparison.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        save_path: Path to save the comparison
        figsize: Figure size
    """
    n_images = len(images)
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        if len(image.shape) == 2:  # Grayscale
            axes[i].imshow(image, cmap='gray')
        else:  # Color
            axes[i].imshow(image)
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Image comparison saved to {save_path}")
    
    plt.show()


def calculate_dataset_statistics(dataset_dir: str) -> Dict[str, Any]:
    """
    Calculate statistics for a dataset.
    
    Args:
        dataset_dir: Directory containing images
        
    Returns:
        Dictionary with dataset statistics
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(dataset_dir).glob(f"*{ext}"))
        image_files.extend(Path(dataset_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        logger.warning(f"No images found in {dataset_dir}")
        return {}
    
    # Collect statistics
    widths = []
    heights = []
    file_sizes = []
    
    for img_path in image_files[:100]:  # Sample first 100 images
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
            file_sizes.append(os.path.getsize(img_path))
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
    
    statistics = {
        'total_images': len(image_files),
        'avg_width': np.mean(widths),
        'avg_height': np.mean(heights),
        'min_width': np.min(widths),
        'max_width': np.max(widths),
        'min_height': np.min(heights),
        'max_height': np.max(heights),
        'avg_file_size_mb': np.mean(file_sizes) / (1024 * 1024),
        'total_size_gb': sum(file_sizes) / (1024 * 1024 * 1024)
    }
    
    logger.info(f"Dataset statistics calculated for {dataset_dir}")
    return statistics


def save_experiment_config(
    config: Dict[str, Any],
    experiment_name: str,
    results_dir: str = "results"
):
    """
    Save experiment configuration for reproducibility.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
        results_dir: Directory to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_copy = config.copy()
    config_copy['experiment_name'] = experiment_name
    config_copy['timestamp'] = timestamp
    
    os.makedirs(results_dir, exist_ok=True)
    config_path = os.path.join(results_dir, f"{experiment_name}_config_{timestamp}.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)
    
    logger.info(f"Experiment config saved to {config_path}")


def create_model_summary(model: torch.nn.Module) -> str:
    """
    Create a text summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary as string
    """
    summary_lines = []
    summary_lines.append("=" * 50)
    summary_lines.append("MODEL ARCHITECTURE SUMMARY")
    summary_lines.append("=" * 50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary_lines.append(f"Total parameters: {total_params:,}")
    summary_lines.append(f"Trainable parameters: {trainable_params:,}")
    summary_lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model structure
    summary_lines.append("\nModel Structure:")
    summary_lines.append("-" * 30)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            summary_lines.append(f"{name}: {module}")
    
    summary_lines.append("=" * 50)
    
    return "\n".join(summary_lines)


def log_gpu_usage():
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            
            logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
    else:
        logger.info("CUDA not available")


def clean_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def validate_config(config_path: str) -> bool:
    """
    Validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'model', 'training', 'paths']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Check required paths exist
        if 'data_root' in config['paths']:
            data_root = config['paths']['data_root']
            if not os.path.exists(data_root):
                logger.warning(f"Data root directory does not exist: {data_root}")
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def create_project_report(
    project_dir: str,
    output_path: str = "project_report.md"
):
    """
    Create a comprehensive project report.
    
    Args:
        project_dir: Project directory
        output_path: Path for the report file
    """
    report_lines = []
    report_lines.append("# Image Colorization Project Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Project structure
    report_lines.append("## Project Structure")
    report_lines.append("```")
    
    for root, dirs, files in os.walk(project_dir):
        level = root.replace(project_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        report_lines.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        
        for file in files[:5]:  # Limit files shown
            report_lines.append(f"{subindent}{file}")
        
        if len(files) > 5:
            report_lines.append(f"{subindent}... and {len(files) - 5} more files")
    
    report_lines.append("```")
    report_lines.append("")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Project report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Utilities module loaded successfully!")
    
    # Test directory setup
    try:
        dirs = setup_project_directories("test_project")
        logger.info(f"Created {len(dirs)} directories")
        
        # Clean up test
        import shutil
        shutil.rmtree("test_project")
        logger.info("Test cleanup completed")
        
    except Exception as e:
        logger.error(f"Error in utilities test: {e}")