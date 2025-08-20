"""
Evaluation module for image colorization models.
Provides metrics calculation and result visualization.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage import color, metrics
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorizationEvaluator:
    """
    Evaluation class for colorization models.
    Computes various metrics and generates visualizations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize metrics
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'mae': []
        }
        
        # Results storage
        self.predictions = []
        self.targets = []
        self.inputs = []
        self.filenames = []
    
    def lab_to_rgb(self, L: torch.Tensor, AB: torch.Tensor) -> np.ndarray:
        """
        Convert LAB tensors to RGB numpy arrays.
        
        Args:
            L: L channel tensor (B, 1, H, W)
            AB: AB channels tensor (B, 2, H, W)
            
        Returns:
            RGB images as numpy array (B, H, W, 3)
        """
        # Denormalize L and AB channels
        L = (L + 1.0) / 2.0 * 100.0  # -1,1 -> 0,100
        AB = AB * 128.0  # -1,1 -> -128,127
        
        batch_size = L.size(0)
        rgb_images = []
        
        for i in range(batch_size):
            # Get individual channels
            l_channel = L[i, 0].cpu().numpy()  # (H, W)
            ab_channels = AB[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)
            
            # Combine LAB channels
            lab_image = np.concatenate([l_channel[..., np.newaxis], ab_channels], axis=2)
            
            # Convert to RGB
            rgb_image = color.lab2rgb(lab_image)
            rgb_image = np.clip(rgb_image, 0, 1)
            
            rgb_images.append(rgb_image)
        
        return np.stack(rgb_images)
    
    def calculate_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def calculate_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Convert to grayscale for SSIM calculation
        if len(pred.shape) == 3:
            pred_gray = color.rgb2gray(pred)
            target_gray = color.rgb2gray(target)
        else:
            pred_gray = pred
            target_gray = target
        
        return metrics.structural_similarity(
            pred_gray, target_gray,
            data_range=1.0,
            win_size=7
        )
    
    def calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS).
        Note: This requires the lpips package to be installed.
        """
        try:
            import lpips
            
            # Initialize LPIPS model (cached after first call)
            if not hasattr(self, 'lpips_model'):
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            
            # Convert to proper format for LPIPS (B, 3, H, W) in range [-1, 1]
            pred_lpips = pred * 2.0 - 1.0  # 0,1 -> -1,1
            target_lpips = target * 2.0 - 1.0
            
            # Calculate LPIPS
            with torch.no_grad():
                distance = self.lpips_model(pred_lpips, target_lpips)
            
            return distance.mean().item()
            
        except ImportError:
            logger.warning("LPIPS package not installed. Skipping LPIPS calculation.")
            return 0.0
    
    def evaluate_batch(
        self, 
        model: nn.Module, 
        L: torch.Tensor, 
        AB_target: torch.Tensor,
        filenames: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of images.
        
        Args:
            model: Trained colorization model
            L: Input L channel (B, 1, H, W)
            AB_target: Target AB channels (B, 2, H, W)
            filenames: List of image filenames
            
        Returns:
            Dictionary of metrics for this batch
        """
        model.eval()
        
        with torch.no_grad():
            # Get model predictions
            AB_pred = model(L.to(self.device))
            
            # Convert to RGB
            pred_rgb = self.lab_to_rgb(L, AB_pred.cpu())
            target_rgb = self.lab_to_rgb(L, AB_target)
            
            # Store results if needed
            if self.eval_config.get('save_predictions', False):
                self.predictions.extend(pred_rgb)
                self.targets.extend(target_rgb)
                self.inputs.extend(L.cpu().numpy())
                self.filenames.extend(filenames)
            
            # Calculate metrics for each image in batch
            batch_metrics = {
                'psnr': [],
                'ssim': [],
                'mse': [],
                'mae': []
            }
            
            for i in range(len(pred_rgb)):
                pred_img = pred_rgb[i]
                target_img = target_rgb[i]
                
                # PSNR
                psnr = self.calculate_psnr(pred_img, target_img)
                batch_metrics['psnr'].append(psnr)
                
                # SSIM
                ssim = self.calculate_ssim(pred_img, target_img)
                batch_metrics['ssim'].append(ssim)
                
                # MSE
                mse = np.mean((pred_img - target_img) ** 2)
                batch_metrics['mse'].append(mse)
                
                # MAE
                mae = np.mean(np.abs(pred_img - target_img))
                batch_metrics['mae'].append(mae)
            
            # Convert to average values
            avg_metrics = {
                key: np.mean(values) for key, values in batch_metrics.items()
            }
            
            return avg_metrics
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        data_loader,
        save_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on entire dataset.
        
        Args:
            model: Trained colorization model
            data_loader: Data loader for evaluation
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary of overall metrics
        """
        model.eval()
        
        all_metrics = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'mae': []
        }
        
        logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch_idx, (L, AB, filenames) in enumerate(data_loader):
                # Evaluate batch
                batch_metrics = self.evaluate_batch(model, L, AB, filenames)
                
                # Accumulate metrics
                for key in all_metrics:
                    all_metrics[key].append(batch_metrics[key])
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Calculate final metrics
        final_metrics = {}
        for key in all_metrics:
            values = np.array(all_metrics[key])
            final_metrics[f'{key}_mean'] = np.mean(values)
            final_metrics[f'{key}_std'] = np.std(values)
        
        # Save metrics
        if save_results:
            self.save_metrics(final_metrics)
        
        # Log results
        logger.info("Evaluation completed!")
        logger.info(f"PSNR: {final_metrics['psnr_mean']:.2f} ± {final_metrics['psnr_std']:.2f}")
        logger.info(f"SSIM: {final_metrics['ssim_mean']:.4f} ± {final_metrics['ssim_std']:.4f}")
        logger.info(f"MSE: {final_metrics['mse_mean']:.6f} ± {final_metrics['mse_std']:.6f}")
        logger.info(f"MAE: {final_metrics['mae_mean']:.6f} ± {final_metrics['mae_std']:.6f}")
        
        return final_metrics
    
    def save_metrics(self, metrics: Dict[str, float], filename: str = "evaluation_metrics.yaml"):
        """Save evaluation metrics to file."""
        results_dir = self.config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        metrics_path = os.path.join(results_dir, filename)
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        logger.info(f"Metrics saved to {metrics_path}")
    
    def create_comparison_grid(
        self, 
        num_samples: int = 8,
        save_path: Optional[str] = None
    ):
        """
        Create a comparison grid showing input, prediction, and target images.
        
        Args:
            num_samples: Number of samples to show
            save_path: Path to save the comparison image
        """
        if not self.predictions:
            logger.warning("No predictions stored. Run evaluation with save_predictions=True first.")
            return
        
        num_samples = min(num_samples, len(self.predictions))
        
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
        
        for i in range(num_samples):
            # Input (grayscale)
            input_gray = self.inputs[i][0]  # L channel
            axes[0, i].imshow(input_gray, cmap='gray')
            axes[0, i].set_title(f'Input\n{self.filenames[i][:15]}...', fontsize=8)
            axes[0, i].axis('off')
            
            # Prediction
            pred_img = self.predictions[i]
            axes[1, i].imshow(pred_img)
            axes[1, i].set_title('Prediction', fontsize=8)
            axes[1, i].axis('off')
            
            # Target
            target_img = self.targets[i]
            axes[2, i].imshow(target_img)
            axes[2, i].set_title('Ground Truth', fontsize=8)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison grid saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of evaluation metrics."""
        if not self.metrics['psnr']:
            logger.warning("No metrics calculated yet. Run evaluation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # PSNR
        axes[0, 0].hist(self.metrics['psnr'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('PSNR Distribution')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        
        # SSIM
        axes[0, 1].hist(self.metrics['ssim'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('SSIM Distribution')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        
        # MSE
        axes[1, 0].hist(self.metrics['mse'], bins=30, alpha=0.7, color='red')
        axes[1, 0].set_title('MSE Distribution')
        axes[1, 0].set_xlabel('MSE')
        axes[1, 0].set_ylabel('Frequency')
        
        # MAE
        axes[1, 1].hist(self.metrics['mae'], bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('MAE Distribution')
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Metrics distribution plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(
        self, 
        model1: nn.Module,
        model2: nn.Module,
        data_loader,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two models side by side.
        
        Args:
            model1: First model (e.g., baseline)
            model2: Second model (e.g., augmented)
            data_loader: Data loader for comparison
            model1_name: Name for first model
            model2_name: Name for second model
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {model1_name} vs {model2_name}")
        
        # Evaluate both models
        metrics1 = self.evaluate_model(model1, data_loader, save_results=False)
        metrics2 = self.evaluate_model(model2, data_loader, save_results=False)
        
        # Create comparison
        comparison = {
            model1_name: metrics1,
            model2_name: metrics2
        }
        
        # Calculate improvements
        improvements = {}
        for metric in ['psnr_mean', 'ssim_mean']:
            diff = metrics2[metric] - metrics1[metric]
            improvement = (diff / metrics1[metric]) * 100
            improvements[metric] = improvement
        
        for metric in ['mse_mean', 'mae_mean']:
            diff = metrics1[metric] - metrics2[metric]  # Lower is better
            improvement = (diff / metrics1[metric]) * 100
            improvements[metric] = improvement
        
        # Log comparison results
        logger.info("\nModel Comparison Results:")
        logger.info("=" * 50)
        
        for metric in ['psnr_mean', 'ssim_mean', 'mse_mean', 'mae_mean']:
            logger.info(f"{metric.upper()}: {model1_name}: {metrics1[metric]:.4f}, "
                       f"{model2_name}: {metrics2[metric]:.4f}, "
                       f"Improvement: {improvements[metric]:+.2f}%")
        
        # Save comparison
        comparison_path = os.path.join(
            self.config['paths']['results_dir'], 
            f"model_comparison_{model1_name}_{model2_name}.yaml"
        )
        
        comparison_data = {
            'comparison': comparison,
            'improvements': improvements
        }
        
        with open(comparison_path, 'w') as f:
            yaml.dump(comparison_data, f, default_flow_style=False)
        
        logger.info(f"Comparison results saved to {comparison_path}")
        
        return comparison_data


def evaluate_single_image(
    model: nn.Module,
    image_path: str,
    output_dir: str = "results/single_evaluation",
    device: Optional[torch.device] = None
):
    """
    Evaluate model on a single image and save results.
    
    Args:
        model: Trained colorization model
        image_path: Path to input image
        output_dir: Directory to save results
        device: Device to run inference on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    L_tensor = preprocessor.prepare_single_image(image_path)
    L_tensor = L_tensor.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        AB_pred = model(L_tensor)
    
    # Convert to RGB
    evaluator = ColorizationEvaluator()
    pred_rgb = evaluator.lab_to_rgb(L_tensor.cpu(), AB_pred.cpu())[0]
    
    # Load original for comparison
    original = Image.open(image_path).convert('RGB')
    original = original.resize((256, 256), Image.Resampling.LANCZOS)
    original_np = np.array(original) / 255.0
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Input (grayscale)
    input_gray = L_tensor[0, 0].cpu().numpy()
    input_gray = (input_gray + 1.0) / 2.0  # Denormalize
    axes[0].imshow(input_gray, cmap='gray')
    axes[0].set_title('Input (Grayscale)')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Colorized (Prediction)')
    axes[1].axis('off')
    
    # Original
    axes[2].imshow(original_np)
    axes[2].set_title('Original (Ground Truth)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    filename = Path(image_path).stem
    result_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    
    # Save individual images
    pred_path = os.path.join(output_dir, f"{filename}_colorized.png")
    pred_pil = Image.fromarray((pred_rgb * 255).astype(np.uint8))
    pred_pil.save(pred_path)
    
    logger.info(f"Single image evaluation completed. Results saved to {output_dir}")
    plt.show()


if __name__ == "__main__":
    # Example usage
    logger.info("Evaluation module loaded successfully!")
    print("This module provides comprehensive evaluation tools for colorization models.")
    print("Use ColorizationEvaluator class for batch evaluation and comparison.")