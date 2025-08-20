"""
Training module for image colorization model.
Handles model training, validation, and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorizationTrainer:
    """
    Main trainer class for colorization model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Colorization model
            train_loader: Training data loader
            val_loader: Validation data loader
            config_path: Path to configuration file
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.device_config = self.config['device']
        self.logging_config = self.config['logging']
        self.paths_config = self.config['paths']
        
        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup mixed precision training
        self.use_amp = self.device_config.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup logging
        self.writer = None
        if self.logging_config.get('use_tensorboard', False):
            log_dir = os.path.join(self.paths_config['logs_dir'], 
                                 f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create directories
        os.makedirs(self.paths_config['models_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.paths_config['models_dir'], 'checkpoints'), exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.device_config.get('use_gpu', True) and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device_config.get('gpu_id', 0)}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.training_config.get('optimizer', 'adam').lower()
        lr = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.training_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Using {optimizer_name.upper()} optimizer with lr={lr}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.training_config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.get('epochs', 100)
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.training_config.get('patience', 10) // 2
            )
        elif scheduler_name == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        if scheduler:
            logger.info(f"Using {scheduler_name} scheduler")
        
        return scheduler
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        loss_name = self.training_config.get('loss_function', 'mse').lower()
        
        if loss_name == 'mse':
            criterion = nn.MSELoss()
        elif loss_name == 'l1':
            criterion = nn.L1Loss()
        elif loss_name == 'smooth_l1':
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        logger.info(f"Using {loss_name.upper()} loss")
        return criterion
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (L, AB, _) in enumerate(progress_bar):
            # Move data to device
            L = L.to(self.device, non_blocking=True)
            AB = AB.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(L)
                    loss = self.criterion(output, AB)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(L)
                loss = self.criterion(output, AB)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # Log batch loss
            if (batch_idx + 1) % self.logging_config.get('log_interval', 10) == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                if self.writer:
                    self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        return epoch_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for L, AB, _ in tqdm(self.val_loader, desc="Validation"):
                L = L.to(self.device, non_blocking=True)
                AB = AB.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(L)
                        loss = self.criterion(output, AB)
                else:
                    output = self.model(L)
                    loss = self.criterion(output, AB)
                
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.val_loader)
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = ""):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{self.current_epoch}{suffix}.pth"
        checkpoint_path = os.path.join(self.paths_config['models_dir'], 'checkpoints', filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.paths_config['models_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def train(self, start_epoch: int = 0) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            start_epoch: Starting epoch (for resuming training)
            
        Returns:
            Training history dictionary
        """
        num_epochs = self.training_config.get('epochs', 100)
        patience = self.training_config.get('patience', 10)
        save_interval = self.logging_config.get('save_interval', 5)
        
        early_stopping_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            if self.writer:
                self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
                self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially restore weights
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_path: str = "config/config.yaml",
    resume_from: Optional[str] = None
) -> Dict[str, list]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config_path: Path to configuration file
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Training history
    """
    trainer = ColorizationTrainer(model, train_loader, val_loader, config_path)
    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        trainer.load_checkpoint(resume_from)
        start_epoch = trainer.current_epoch + 1
    
    return trainer.train(start_epoch)


if __name__ == "__main__":
    # Example usage
    logger.info("Training module loaded successfully!")
    print("This module provides the ColorizationTrainer class for training colorization models.")
    print("Use it in conjunction with your model and data loaders.")