"""Models module for neural classification."""
from .cnn_branch import SpectrogramCNN, create_cnn_model
from .gnn_branch import ConnectivityGNN, create_gnn_model
from .hybrid_fusion import HybridClassifier, create_hybrid_model
from .attention import SelfAttention, ChannelAttention, FusionAttention

__all__ = [
    'SpectrogramCNN', 'create_cnn_model',
    'ConnectivityGNN', 'create_gnn_model',
    'HybridClassifier', 'create_hybrid_model',
    'SelfAttention', 'ChannelAttention', 'FusionAttention'
]
