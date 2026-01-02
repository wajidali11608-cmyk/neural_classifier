"""
CNN Branch for Spectrogram Classification

2D Convolutional Neural Network for extracting features from
time-frequency spectrograms of neural signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .attention import ChannelAttention, CBAM


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm, activation, and optional attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.conv = nn.Sequential(*layers)
        
        self.attention = CBAM(out_channels) if use_attention else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = ChannelAttention(out_channels) if use_attention else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.attention is not None:
            out = self.attention(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpectrogramCNN(nn.Module):
    """
    CNN model for spectrogram-based neural signal classification.
    
    Architecture:
    - Multiple convolutional blocks with optional attention
    - Global average pooling
    - Fully connected layers for embedding/classification
    
    Args:
        in_channels: Number of input channels (1 for single spectrogram)
        n_classes: Number of output classes
        base_filters: Base number of filters (doubled each block)
        n_blocks: Number of convolutional blocks
        fc_hidden_dim: Hidden dimension of FC layers
        embedding_dim: Dimension of embedding output
        dropout: Dropout probability
        use_attention: Whether to use attention modules
        use_residual: Use residual connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 3,
        base_filters: int = 32,
        n_blocks: int = 4,
        fc_hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.5,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(n_blocks):
            out_channels = base_filters * (2 ** i)
            
            if use_residual:
                block = ResidualBlock(
                    current_channels, out_channels,
                    stride=2 if i > 0 else 1,
                    use_attention=use_attention
                )
            else:
                block = nn.Sequential(
                    ConvBlock(
                        current_channels, out_channels,
                        use_attention=use_attention
                    ),
                    nn.MaxPool2d(2, 2) if i > 0 else nn.Identity()
                )
            
            self.conv_blocks.append(block)
            current_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(current_channels, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
        # Store last feature map for Grad-CAM
        self.feature_maps = None
        self.gradients = None
        
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram, shape (batch, channels, freq, time)
            return_embedding: If True, return embedding instead of logits
            
        Returns:
            If return_embedding: embedding of shape (batch, embedding_dim)
            Else: logits of shape (batch, n_classes)
        """
        # Convolutional feature extraction
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
        
        # Store for Grad-CAM
        self.feature_maps = x
        
        # Global pooling -> flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers -> embedding
        embedding = self.fc(x)
        
        if return_embedding:
            return embedding
        
        # Classification
        logits = self.classifier(embedding)
        
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding without classification head."""
        return self.forward(x, return_embedding=True)
    
    def get_feature_maps(self) -> Optional[torch.Tensor]:
        """Return stored feature maps for visualization."""
        return self.feature_maps
    
    def register_hooks(self):
        """Register hooks for Grad-CAM."""
        def forward_hook(module, input, output):
            self.feature_maps = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register on last conv block
        self.conv_blocks[-1].register_forward_hook(forward_hook)
        self.conv_blocks[-1].register_full_backward_hook(backward_hook)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for faster inference.
    
    Uses depthwise separable convolutions.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 3,
        base_filters: int = 16,
        embedding_dim: int = 64
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, base_filters, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            self._make_dsconv_block(base_filters, base_filters * 2, 2),
            self._make_dsconv_block(base_filters * 2, base_filters * 4, 2),
            self._make_dsconv_block(base_filters * 4, base_filters * 8, 2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(base_filters * 8, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
    def _make_dsconv_block(self, in_ch: int, out_ch: int, stride: int = 1):
        """Depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        
        if return_embedding:
            return embedding
        
        return self.classifier(embedding)


def create_cnn_model(
    in_channels: int = 1,
    n_classes: int = 3,
    model_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create CNN model.
    
    Args:
        in_channels: Input channels
        n_classes: Number of classes
        model_type: 'standard' or 'lightweight'
        **kwargs: Additional model arguments
        
    Returns:
        CNN model
    """
    if model_type == 'standard':
        return SpectrogramCNN(in_channels, n_classes, **kwargs)
    elif model_type == 'lightweight':
        return LightweightCNN(in_channels, n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
