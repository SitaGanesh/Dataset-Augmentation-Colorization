"""
Model architecture for image colorization.
Implements U-Net based architecture for converting grayscale to color images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import yaml
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetEncoder(nn.Module):
    """U-Net encoder with skip connections."""
    
    def __init__(self, in_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super(UNetEncoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        return x, skip_connections


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections."""
    
    def __init__(self, features: List[int] = [512, 256, 128, 64], out_channels: int = 2):
        super(UNetDecoder, self).__init__()
        self.ups = nn.ModuleList()
        
        # Decoder path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # Double conv
        
        return self.final_conv(x)


class UNet(nn.Module):
    """
    Complete U-Net architecture for image colorization.
    Takes grayscale (L channel) as input and predicts AB channels.
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 2, 
        features: List[int] = [64, 128, 256, 512],
        dropout: float = 0.2
    ):
        super(UNet, self).__init__()
        
        self.encoder = UNetEncoder(in_channels, features)
        self.decoder = UNetDecoder(features, out_channels)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout)
    
    def forward(self, x):
        # Encoder
        x, skip_connections = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder(x, skip_connections)
        
        # Apply tanh to constrain output to [-1, 1] range (for AB channels)
        x = torch.tanh(x)
        
        return x


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet backbone for improved feature extraction.
    """
    
    def __init__(
        self, 
        backbone: str = "resnet34",
        in_channels: int = 1,
        out_channels: int = 2,
        pretrained: bool = True
    ):
        super(ResNetUNet, self).__init__()
        
        # Load pretrained ResNet backbone
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first layer for grayscale input
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract encoder layers
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # Decoder layers
        self.decoder4 = self._make_decoder_layer(encoder_channels[4], encoder_channels[3])
        self.decoder3 = self._make_decoder_layer(encoder_channels[3], encoder_channels[2])
        self.decoder2 = self._make_decoder_layer(encoder_channels[2], encoder_channels[1])
        self.decoder1 = self._make_decoder_layer(encoder_channels[1], encoder_channels[0])
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def _make_decoder_layer(self, in_channels: int, out_channels: int):
        """Create a decoder layer."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)  # 64 channels
        e1 = self.encoder1(e0)  # 64 channels
        e2 = self.encoder2(e1)  # 128 channels
        e3 = self.encoder3(e2)  # 256 channels
        e4 = self.encoder4(e3)  # 512 channels
        
        # Decoder with skip connections
        d4 = self.decoder4[0](e4)  # Upsample
        d4 = torch.cat([d4, e3], dim=1)  # Skip connection
        d4 = self.decoder4[1:](d4)  # Conv layers
        
        d3 = self.decoder3[0](d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3[1:](d3)
        
        d2 = self.decoder2[0](d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2[1:](d2)
        
        d1 = self.decoder1[0](d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.decoder1[1:](d1)
        
        # Final output
        output = self.final_conv(d1)
        
        # Upsample to original size if needed
        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        return output


class ColorizationModel(nn.Module):
    """
    Main colorization model that can use different architectures.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        super(ColorizationModel, self).__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        
        # Initialize model based on architecture
        if model_config['architecture'] == 'unet':
            self.model = UNet(
                in_channels=model_config['input_channels'],
                out_channels=model_config['output_channels'],
                features=[64, 128, 256, 512],
                dropout=model_config['unet']['dropout']
            )
        elif model_config['architecture'] == 'resnet_unet':
            self.model = ResNetUNet(
                backbone=model_config['backbone'],
                in_channels=model_config['input_channels'],
                out_channels=model_config['output_channels'],
                pretrained=model_config['pretrained']
            )
        else:
            raise ValueError(f"Unsupported architecture: {model_config['architecture']}")
        
        self.architecture = model_config['architecture']
        logger.info(f"Initialized {self.architecture} model")
    
    def forward(self, x):
        return self.model(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self):
        """Get model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features for better colorization quality.
    """
    
    def __init__(self, layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_2']):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Layer mapping
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2", 
            '15': "relu3_2"
        }
        
        self.target_layers = layers
        self.loss_fn = nn.MSELoss()
    
    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor):
        """
        Compute perceptual loss.
        
        Args:
            input_img: Generated RGB image (B, 3, H, W)
            target_img: Target RGB image (B, 3, H, W)
        """
        input_features = self.extract_features(input_img)
        target_features = self.extract_features(target_img)
        
        loss = 0
        for layer in self.target_layers:
            if layer in input_features and layer in target_features:
                loss += self.loss_fn(input_features[layer], target_features[layer])
        
        return loss
    
    def extract_features(self, img: torch.Tensor):
        """Extract VGG features."""
        features = {}
        x = img
        
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                features[self.layer_name_mapping[name]] = x
        
        return features


def create_model(config_path: str = "config/config.yaml") -> ColorizationModel:
    """
    Factory function to create a colorization model.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized colorization model
    """
    model = ColorizationModel(config_path)
    
    # Print model info
    num_params = model.count_parameters()
    model_size = model.get_model_size()
    
    logger.info(f"Model created with {num_params:,} trainable parameters")
    logger.info(f"Model size: {model_size:.2f} MB")
    
    return model


def initialize_weights(model: nn.Module):
    """Initialize model weights using He initialization."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Test model creation
    try:
        # Create model
        model = create_model()
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 256, 256)  # Batch of 1, grayscale, 256x256
        output = model(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful!")
        
        # Test perceptual loss
        perceptual_loss = PerceptualLoss()
        dummy_rgb_input = torch.randn(1, 3, 256, 256)
        dummy_rgb_target = torch.randn(1, 3, 256, 256)
        
        loss_value = perceptual_loss(dummy_rgb_input, dummy_rgb_target)
        print(f"Perceptual loss test: {loss_value.item():.4f}")
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        print("Please ensure config files are properly set up.")