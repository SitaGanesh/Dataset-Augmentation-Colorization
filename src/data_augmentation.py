# src/data_augmentation.py
"""
Data augmentation module for image colorization project.
Implements various augmentation techniques to improve model robustness.
"""

import os
import sys
import random
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
import albumentations as A
from typing import List, Tuple, Optional
import yaml
from tqdm import tqdm
import albumentations as A


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentedColorDataset(Dataset):
    """
    Extended dataset class with augmentation support.
    Applies various augmentation techniques while preserving color consistency.
    """
    
    def __init__(
        self,
        image_dir: str,
        augmentation_config_path: str = "config/augmentation_config.yaml",
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True
    ):
        """
        Initialize augmented dataset.
        
        Args:
            image_dir: Directory containing color images
            augmentation_config_path: Path to augmentation configuration
            image_size: Target image size
            is_training: Whether to apply augmentations
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_training = is_training
        
        # Load augmentation configuration
        with open(augmentation_config_path, 'r') as f:
            self.aug_config = yaml.safe_load(f)
        
        # Get image files
        import os
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
        logger.info(f"Initialized augmented dataset with {len(self.image_files)} images")
        logger.info(f"Augmentation enabled: {self.is_training and self.aug_config['augmentation']['enabled']}")
    
    def _create_augmentation_pipeline(self):
        """Create albumentations augmentation pipeline."""
        if not self.is_training or not self.aug_config['augmentation']['enabled']:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        aug_config = self.aug_config['augmentation']
        strategy = self.aug_config['strategies'][self.aug_config['current_strategy']]
        
        augmentations = []
        
        # Resize first
        augmentations.append(A.Resize(height=self.image_size[0], width=self.image_size[1]))
        
        # Geometric augmentations
        if random.random() < strategy['geometric_prob']:
            geometric_augs = self._get_geometric_augmentations(aug_config['geometric'])
            augmentations.extend(geometric_augs)
        
        # Photometric augmentations  
        if random.random() < strategy['photometric_prob']:
            photometric_augs = self._get_photometric_augmentations(aug_config['photometric'])
            augmentations.extend(photometric_augs)
        
        # Advanced augmentations
        if random.random() < strategy['advanced_prob']:
            advanced_augs = self._get_advanced_augmentations(aug_config['advanced'])
            augmentations.extend(advanced_augs)
        
        # Color augmentations (for target images)
        color_augs = self._get_color_augmentations(aug_config['color_augmentation'])
        augmentations.extend(color_augs)
        
        # Normalization
        augmentations.append(A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        return A.Compose(augmentations)
    
    def _get_geometric_augmentations(self, config: Dict[str, Any]) -> List:
        """Get geometric augmentation transforms."""
        augs = []
        
        if config['horizontal_flip']['enabled']:
            augs.append(A.HorizontalFlip(p=config['horizontal_flip']['probability']))
        
        if config['vertical_flip']['enabled']:
            augs.append(A.VerticalFlip(p=config['vertical_flip']['probability']))
        
        if config['rotation']['enabled']:
            augs.append(A.Rotate(
                limit=config['rotation']['angle_range'],
                p=config['rotation']['probability'],
                border_mode=cv2.BORDER_REFLECT
            ))
        
        if config['scale']['enabled']:
            augs.append(A.RandomScale(
                scale_limit=config['scale']['scale_range'],
                p=config['scale']['probability']
            ))
        
        if config['translation']['enabled']:
            translate_percent = config['translation']['translate_percent']
            augs.append(A.ShiftScaleRotate(
                shift_limit=translate_percent,
                scale_limit=0,
                rotate_limit=0,
                p=config['translation']['probability'],
                border_mode=cv2.BORDER_REFLECT
            ))
        
        if config['shear']['enabled']:
            augs.append(A.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0,
                rotate_limit=0,
                shear_limit=config['shear']['shear_range'],
                p=config['shear']['probability'],
                border_mode=cv2.BORDER_REFLECT
            ))
        
        return augs
    
    def _get_photometric_augmentations(self, config):
        """Get photometric augmentations."""
        augs = []
        
        # Use ColorJitter instead of RandomBrightness
        if config['brightness_contrast']['enabled']:
            augs.append(A.ColorJitter(
                brightness=config['brightness_contrast']['brightness_limit'],
                contrast=config['brightness_contrast']['contrast_limit'],
                p=config['brightness_contrast']['probability']
            ))
        
        if config['hue_saturation']['enabled']:
            augs.append(A.HueSaturationValue(
                hue_shift_limit=config['hue_saturation']['hue_shift_limit'],
                sat_shift_limit=config['hue_saturation']['sat_shift_limit'],
                val_shift_limit=config['hue_saturation']['val_shift_limit'],
                p=config['hue_saturation']['probability']
            ))
        
        if config['gamma_transform']['enabled']:
            augs.append(A.RandomGamma(
                gamma_limit=config['gamma_transform']['gamma_limit'],
                p=config['gamma_transform']['probability']
            ))
        
        if config['gaussian_noise']['enabled']:
            augs.append(A.GaussNoise(
                var_limit=config['gaussian_noise']['var_limit'],
                p=config['gaussian_noise']['probability']
            ))
        
        return augs

    
    def _get_advanced_augmentations(self, config: Dict[str, Any]) -> List:
        """Get advanced augmentation transforms."""
        augs = []
        
        if config['elastic_transform']['enabled']:
            augs.append(A.ElasticTransform(
                alpha=config['elastic_transform']['alpha'],
                sigma=config['elastic_transform']['sigma'],
                p=config['elastic_transform']['probability'],
                border_mode=cv2.BORDER_REFLECT
            ))
        
        if config['grid_distortion']['enabled']:
            augs.append(A.GridDistortion(
                num_steps=config['grid_distortion']['num_steps'],
                distort_limit=config['grid_distortion']['distort_limit'],
                p=config['grid_distortion']['probability'],
                border_mode=cv2.BORDER_REFLECT
            ))
        
        if config['optical_distortion']['enabled']:
            augs.append(A.OpticalDistortion(
                distort_limit=config['optical_distortion']['distort_limit'],
                shift_limit=config['optical_distortion']['shift_limit'],
                p=config['optical_distortion']['probability'],
                border_mode=cv2.BORDER_REFLECT
            ))
        
        if config['coarse_dropout']['enabled']:
            augs.append(A.CoarseDropout(
                max_holes=config['coarse_dropout']['max_holes'],
                max_height=config['coarse_dropout']['max_height'],
                max_width=config['coarse_dropout']['max_width'],
                p=config['coarse_dropout']['probability']
            ))
        
        return augs
    
    def _get_color_augmentations(self, config: Dict[str, Any]) -> List:
        """Get color-specific augmentation transforms."""
        augs = []
        
        if config['color_jitter']['enabled']:
            augs.append(A.ColorJitter(
                brightness=config['color_jitter']['brightness'],
                contrast=config['color_jitter']['contrast'],
                saturation=config['color_jitter']['saturation'],
                hue=config['color_jitter']['hue'],
                p=config['color_jitter']['probability']
            ))
        
        if config['solarization']['enabled']:
            augs.append(A.Solarize(
                threshold=config['solarization']['threshold'],
                p=config['solarization']['probability']
            ))
        
        return augs
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get augmented sample from dataset."""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        augmented = self.augmentation_pipeline(image=image)
        image_aug = augmented['image']
        
        # Convert to LAB color space
        if isinstance(image_aug, np.ndarray):
            # Denormalize for color space conversion
            image_denorm = (image_aug * 0.5 + 0.5) * 255
            image_denorm = image_denorm.astype(np.uint8)
            
            # Convert to LAB
            from skimage import color
            lab_image = color.rgb2lab(image_denorm).astype(np.float32)
            
            # Normalize LAB values
            lab_image[:, :, 0] = lab_image[:, :, 0] / 100.0  # L: 0-100 -> 0-1
            lab_image[:, :, 1:] = lab_image[:, :, 1:] / 128.0  # AB: -128-127 -> -1-1
            
            # Separate channels
            L = lab_image[:, :, 0]  # Grayscale input
            AB = lab_image[:, :, 1:]  # Color target
            
            # Convert to tensors
            L = torch.from_numpy(L).unsqueeze(0)  # (1, H, W)
            AB = torch.from_numpy(AB.transpose(2, 0, 1))  # (2, H, W)
            
            # Normalize for training
            L = (L - 0.5) / 0.5  # L channel normalization
        
        return L, AB, self.image_files[idx]


class AugmentationVisualizer:
    """
    Utility class for visualizing augmentation effects.
    """
    
    def __init__(self, config_path: str = "config/augmentation_config.yaml"):
        """Initialize visualizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def preview_augmentations(
        self, 
        image_path: str, 
        num_samples: int = 5,
        save_path: str = None
    ):
        """
        Preview augmentation effects on a single image.
        
        Args:
            image_path: Path to the input image
            num_samples: Number of augmented samples to generate
            save_path: Path to save preview images
        """
        import matplotlib.pyplot as plt
        
        # Load original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Create dataset with augmentation
        dataset = AugmentedColorDataset(
            os.path.dirname(image_path),
            is_training=True
        )
        
        # Generate augmented samples
        fig, axes = plt.subplots(2, num_samples + 1, figsize=(20, 8))
        
        # Show original
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        
        # Convert original to grayscale for comparison
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        axes[1, 0].imshow(gray_original, cmap='gray')
        axes[1, 0].set_title("Original (Grayscale)")
        axes[1, 0].axis('off')
        
        # Generate and show augmented samples
        filename = os.path.basename(image_path)
        file_idx = dataset.image_files.index(filename)
        
        for i in range(num_samples):
            L, AB, _ = dataset[file_idx]
            
            # Convert back to RGB for visualization
            from src.data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            
            # Denormalize
            L_denorm = (L + 1.0) / 2.0  # -1,1 -> 0,1
            AB_denorm = AB  # Already in -1,1 range
            
            # Convert to RGB
            rgb_images = preprocessor.lab_to_rgb(
                L_denorm.unsqueeze(0), 
                AB_denorm.unsqueeze(0)
            )
            
            rgb_image = rgb_images[0]
            gray_image = L_denorm.squeeze().numpy()
            
            # Show color version
            axes[0, i + 1].imshow(rgb_image)
            axes[0, i + 1].set_title(f"Augmented {i+1}")
            axes[0, i + 1].axis('off')
            
            # Show grayscale version (model input)
            axes[1, i + 1].imshow(gray_image, cmap='gray')
            axes[1, i + 1].set_title(f"Input {i+1}")
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Augmentation preview saved to {save_path}")
        
        plt.show()
    
    def analyze_augmentation_statistics(
        self, 
        dataset_dir: str, 
        num_samples: int = 100
    ):
        """
        Analyze statistical properties of augmented dataset.
        
        Args:
            dataset_dir: Directory containing images
            num_samples: Number of samples to analyze
        """
        dataset = AugmentedColorDataset(dataset_dir, is_training=True)
        
        # Collect statistics
        L_values = []
        AB_values = []
        
        for i in range(min(num_samples, len(dataset))):
            L, AB, _ = dataset[i]
            L_values.append(L.numpy().flatten())
            AB_values.append(AB.numpy().flatten())
        
        L_all = np.concatenate(L_values)
        AB_all = np.concatenate(AB_values)
        
        # Print statistics
        print("Augmented Dataset Statistics:")
        print(f"L channel - Mean: {L_all.mean():.4f}, Std: {L_all.std():.4f}")
        print(f"L channel - Min: {L_all.min():.4f}, Max: {L_all.max():.4f}")
        print(f"AB channels - Mean: {AB_all.mean():.4f}, Std: {AB_all.std():.4f}")
        print(f"AB channels - Min: {AB_all.min():.4f}, Max: {AB_all.max():.4f}")
        
        # Plot histograms
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(L_all, bins=50, alpha=0.7, color='gray')
        axes[0].set_title('L Channel Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(AB_all[::2], bins=50, alpha=0.7, color='red', label='A channel')
        axes[1].hist(AB_all[1::2], bins=50, alpha=0.7, color='blue', label='B channel')
        axes[1].set_title('AB Channels Distribution')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # Combined plot
        axes[2].hist(L_all, bins=50, alpha=0.5, color='gray', label='L')
        axes[2].hist(AB_all, bins=50, alpha=0.5, color='purple', label='AB')
        axes[2].set_title('All Channels Distribution')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()


def create_augmented_dataset(
    source_dir: str,
    output_dir: str,
    augmentation_config_path: str,
    multiplier: int = 3
):
    """
    Create an augmented dataset by generating multiple variations of each image.
    
    Args:
        source_dir: Source directory with original images
        output_dir: Output directory for augmented images
        augmentation_config_path: Path to augmentation configuration
        multiplier: Number of augmented versions per original image
    """
    import os
    import shutil
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy original images
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(source_dir, filename),
                os.path.join(output_dir, filename)
            )
    
    # Generate augmented versions
    dataset = AugmentedColorDataset(
        source_dir,
        augmentation_config_path,
        is_training=True
    )
    
    for i in tqdm(range(len(dataset)), desc="Generating augmented images"):
        original_filename = dataset.image_files[i]
        name, ext = os.path.splitext(original_filename)
        
        for j in range(multiplier):
            # Get augmented sample
            L, AB, _ = dataset[i]
            
            # Convert back to RGB
            from src.data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            
            L_denorm = (L + 1.0) / 2.0
            rgb_images = preprocessor.lab_to_rgb(
                L_denorm.unsqueeze(0), 
                AB.unsqueeze(0)
            )
            
            # Save augmented image
            rgb_image = (rgb_images[0] * 255).astype(np.uint8)
            aug_filename = f"{name}_aug_{j}{ext}"
            
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(os.path.join(output_dir, aug_filename))
    
    logger.info(f"Created augmented dataset in {output_dir}")
    logger.info(f"Original images: {len(dataset)}")
    logger.info(f"Total images: {len(dataset) * (multiplier + 1)}")


if __name__ == "__main__":
    # Example usage
    config_path = "config/augmentation_config.yaml"
    
    # Test augmentation pipeline
    try:
        # Preview augmentations (if you have a test image)
        visualizer = AugmentationVisualizer(config_path)
        # visualizer.preview_augmentations("path/to/test/image.jpg")
        
        logger.info("Augmentation module loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error testing augmentation: {e}")