"""
Attention Mechanisms for Neural Classification

Implements self-attention, channel attention, and cross-modal fusion
attention for improved feature weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Standard transformer-style attention for sequence/spatial features.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            output: Attended features
            attn_weights: Attention weights for visualization
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x, attn
    

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    
    Learns to weight different feature channels based on global information.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, channels, height, width)
            
        Returns:
            Attention-weighted features
        """
        B, C, H, W = x.shape
        
        # Global pooling
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        
        # Channel attention weights
        avg_attn = self.fc(avg_out)
        max_attn = self.fc(max_out)
        
        attn = (avg_attn + max_attn) / 2
        attn = attn.view(B, C, 1, 1)
        
        return x * attn


class SpatialAttention(nn.Module):
    """
    Spatial attention for highlighting important regions.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, channels, height, width)
            
        Returns:
            Spatially attended features
        """
        # Aggregate channel information
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        
        concat = torch.cat([avg_out, max_out], dim=1)
        
        attn = self.sigmoid(self.conv(concat))
        
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel and spatial attention sequentially.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel: int = 7
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FusionAttention(nn.Module):
    """
    Cross-modal attention for fusing CNN and GNN features.
    
    Learns to weight contributions from different modalities.
    """
    
    def __init__(
        self,
        cnn_dim: int,
        gnn_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cnn_dim = cnn_dim
        self.gnn_dim = gnn_dim
        
        # Project to common dimension
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # Cross-attention: CNN attends to GNN
        self.cross_attn_cnn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention: GNN attends to CNN
        self.cross_attn_gnn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Layer norms
        self.norm_cnn = nn.LayerNorm(hidden_dim)
        self.norm_gnn = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        cnn_features: torch.Tensor,
        gnn_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_features: CNN output, shape (batch, cnn_dim)
            gnn_features: GNN output, shape (batch, gnn_dim)
            
        Returns:
            fused_features: Combined features
            gate_weights: Weights for each modality (for interpretation)
        """
        B = cnn_features.shape[0]
        
        # Project to common space
        cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)  # (B, 1, hidden)
        gnn_proj = self.gnn_proj(gnn_features).unsqueeze(1)  # (B, 1, hidden)
        
        # Cross-attention
        cnn_attended, _ = self.cross_attn_cnn(cnn_proj, gnn_proj, gnn_proj)
        gnn_attended, _ = self.cross_attn_gnn(gnn_proj, cnn_proj, cnn_proj)
        
        # Add residual and normalize
        cnn_out = self.norm_cnn(cnn_proj + cnn_attended).squeeze(1)
        gnn_out = self.norm_gnn(gnn_proj + gnn_attended).squeeze(1)
        
        # Compute gating weights
        concat = torch.cat([cnn_out, gnn_out], dim=-1)
        gate_weights = self.gate(concat)  # (B, 2)
        
        # Weighted fusion
        fused = gate_weights[:, 0:1] * cnn_out + gate_weights[:, 1:2] * gnn_out
        
        return fused, gate_weights


class GraphAttentionPooling(nn.Module):
    """
    Attention-based global graph pooling.
    
    Learns to weight node contributions to graph-level representation.
    """
    
    def __init__(self, in_features: int, hidden_features: int = 64):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1, bias=False)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features, shape (n_nodes, features)
            batch: Batch assignment for each node
            
        Returns:
            pooled: Graph-level features
            attn_weights: Node attention weights
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)
        
        if batch is None:
            # Single graph
            attn_weights = F.softmax(scores, dim=0)
            pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=0, keepdim=True)
        else:
            # Batched graphs
            from torch_geometric.utils import softmax, scatter
            attn_weights = softmax(scores, batch)
            pooled = scatter(attn_weights.unsqueeze(-1) * x, batch, dim=0, reduce='sum')
        
        return pooled, attn_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention for time-series features.
    
    Learns to focus on important time windows.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64
    ):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Temporal features, shape (batch, time, features)
            
        Returns:
            pooled: Attended features, shape (batch, features)
            attn_weights: Temporal attention, shape (batch, time)
        """
        # Compute attention
        scores = self.attention(x).squeeze(-1)  # (batch, time)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        
        return pooled, attn_weights
