"""
Data preprocessing utilities for image colorization project.
Handles image loading, color space conversion, and dataset preparation.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage import color
import yaml
from typing import Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColorDataset(Dataset):
    """
    Dataset class for loading and preprocessing color images for colorization.
    Converts RGB images to LAB color space and separates L channel (input) 
    from AB channels (target).
    """
    
    def __init__(
        self, 
        image_dir: str, 
        transform=None, 
        target_transform=None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing color images
            transform: Transforms for input (L channel)
            target_transform: Transforms for target (AB channels)
            image_size: Target image size (height, width)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert RGB to LAB
        lab_image = color.rgb2lab(image_np).astype(np.float32)
        
        # Normalize LAB values
        # L: 0-100 -> 0-1, AB: -128-127 -> -1-1
        lab_image[:, :, 0] = lab_image[:, :, 0] / 100.0
        lab_image[:, :, 1:] = lab_image[:, :, 1:] / 128.0
        
        # Separate L and AB channels
        L = lab_image[:, :, 0]  # Grayscale input
        AB = lab_image[:, :, 1:]  # Color target
        
        # Convert to tensors and rearrange dimensions
        L = torch.from_numpy(L).unsqueeze(0)  # (1, H, W)
        AB = torch.from_numpy(AB.transpose(2, 0, 1))  # (2, H, W)
        
        # Apply transforms
        if self.transform:
            L = self.transform(L)
        
        if self.target_transform:
            AB = self.target_transform(AB)
        
        return L, AB, self.image_files[idx]


class DataPreprocessor:
    """
    Main class for data preprocessing operations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.image_size = tuple(self.data_config['input_size'])
        
    def get_transforms(self, is_training: bool = True):
        """
        Get data transforms for training/validation.
        
        Args:
            is_training: Whether to apply training augmentations
            
        Returns:
            Tuple of (input_transform, target_transform)
        """
        if is_training:
            input_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=[0.5], std=[0.5])  # L channel normalization
            ])
            
            target_transform = transforms.Compose([
                transforms.Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])  # AB channels
            ])
        else:
            input_transform = transforms.Compose([
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            target_transform = transforms.Compose([
                transforms.Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])
            ])
        
        return input_transform, target_transform
    
    def create_dataloaders(
        self, 
        train_dir: str, 
        val_dir: str, 
        test_dir: str = None
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory  
            test_dir: Test data directory (optional)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Get transforms
        train_input_transform, train_target_transform = self.get_transforms(True)
        val_input_transform, val_target_transform = self.get_transforms(False)
        
        # Create datasets
        train_dataset = ColorDataset(
            train_dir, 
            transform=train_input_transform,
            target_transform=train_target_transform,
            image_size=self.image_size
        )
        
        val_dataset = ColorDataset(
            val_dir,
            transform=val_input_transform, 
            target_transform=val_target_transform,
            image_size=self.image_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=True
        )
        
        test_loader = None
        if test_dir:
            test_dataset = ColorDataset(
                test_dir,
                transform=val_input_transform,
                target_transform=val_target_transform,
                image_size=self.image_size
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.data_config['batch_size'],
                shuffle=False,
                num_workers=self.data_config['num_workers'],
                pin_memory=True
            )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train: {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_loader)} batches")
        if test_loader:
            logger.info(f"  Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def lab_to_rgb(self, L: torch.Tensor, AB: torch.Tensor) -> np.ndarray:
        """
        Convert LAB tensors back to RGB numpy array.
        
        Args:
            L: L channel tensor (B, 1, H, W)
            AB: AB channels tensor (B, 2, H, W)
            
        Returns:
            RGB image as numpy array (B, H, W, 3)
        """
        # Denormalize
        L = L * 100.0  # 0-1 -> 0-100
        AB = AB * 128.0  # -1-1 -> -128-127
        
        # Combine LAB channels
        batch_size = L.size(0)
        lab_images = []
        
        for i in range(batch_size):
            l_channel = L[i, 0].cpu().numpy()  # (H, W)
            ab_channels = AB[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)
            
            lab_image = np.concatenate([l_channel[..., np.newaxis], ab_channels], axis=2)
            rgb_image = color.lab2rgb(lab_image)
            rgb_image = np.clip(rgb_image, 0, 1)
            
            lab_images.append(rgb_image)
        
        return np.stack(lab_images)
    
    def prepare_single_image(self, image_path: str) -> torch.Tensor:
        """
        Prepare a single image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed L channel tensor
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to LAB
        image_np = np.array(image)
        lab_image = color.rgb2lab(image_np).astype(np.float32)
        
        # Normalize L channel
        L = lab_image[:, :, 0] / 100.0
        
        # Convert to tensor
        L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Normalize
        L_tensor = (L_tensor - 0.5) / 0.5
        
        return L_tensor


def split_dataset(
    source_dir: str, 
    train_dir: str, 
    val_dir: str, 
    test_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05
):
    """
    Split a dataset into train/validation/test sets.
    
    Args:
        source_dir: Source directory containing all images
        train_dir: Output training directory
        val_dir: Output validation directory
        test_dir: Output test directory
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        test_ratio: Ratio of test samples
    """
    import shutil
    import random
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_idx = int(total_files * train_ratio)
    val_idx = int(total_files * (train_ratio + val_ratio))
    
    # Split files
    train_files = image_files[:train_idx]
    val_files = image_files[train_idx:val_idx]
    test_files = image_files[val_idx:]
    
    # Copy files to respective directories
    for file in train_files:
        shutil.copy2(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.copy2(os.path.join(source_dir, file), os.path.join(val_dir, file))
    
    for file in test_files:
        shutil.copy2(os.path.join(source_dir, file), os.path.join(test_dir, file))
    
    logger.info(f"Dataset split completed:")
    logger.info(f"  Train: {len(train_files)} images")
    logger.info(f"  Val: {len(val_files)} images")
    logger.info(f"  Test: {len(test_files)} images")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Create data loaders (assuming data directories exist)
    try:
        train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
            "data/processed/train",
            "data/processed/val", 
            "data/processed/test"
        )
        
        # Test loading a batch
        for L, AB, filenames in train_loader:
            print(f"Input shape (L): {L.shape}")
            print(f"Target shape (AB): {AB.shape}")
            print(f"Batch filenames: {filenames[:3]}")
            break
            
    except FileNotFoundError:
        logger.warning("Data directories not found. Please prepare your dataset first.")