"""
Hybrid Fusion Model for Neural Classification

Combines CNN and GNN branches with attention-based fusion
for comprehensive analysis of both spectral and connectivity features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
from torch_geometric.data import Data, Batch

from .cnn_branch import SpectrogramCNN
from .gnn_branch import ConnectivityGNN
from .attention import FusionAttention


class HybridClassifier(nn.Module):
    """
    Hybrid model combining CNN and GNN for neural signal classification.
    
    Processes both:
    - Spectrograms through CNN for time-frequency patterns
    - Connectivity graphs through GNN for network topology
    
    Uses attention-based fusion to combine modality embeddings.
    
    Args:
        cnn_config: Configuration for CNN branch
        gnn_config: Configuration for GNN branch
        n_classes: Number of output classes
        fusion_type: 'concat', 'attention', or 'gated'
        fc_hidden_dims: Hidden dimensions for FC layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        cnn_config: Optional[Dict] = None,
        gnn_config: Optional[Dict] = None,
        n_classes: int = 3,
        fusion_type: str = 'attention',
        fc_hidden_dims: list = None,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.fusion_type = fusion_type
        
        if fc_hidden_dims is None:
            fc_hidden_dims = [256, 128]
        
        # Default configs
        cnn_config = cnn_config or {}
        gnn_config = gnn_config or {}
        
        # CNN branch - don't include classifier, we'll fuse embeddings
        cnn_embedding_dim = cnn_config.get('embedding_dim', 128)
        self.cnn = SpectrogramCNN(
            in_channels=cnn_config.get('in_channels', 1),
            n_classes=n_classes,
            base_filters=cnn_config.get('base_filters', 32),
            n_blocks=cnn_config.get('n_blocks', 4),
            fc_hidden_dim=cnn_config.get('fc_hidden_dim', 256),
            embedding_dim=cnn_embedding_dim,
            dropout=cnn_config.get('dropout', 0.5),
            use_attention=cnn_config.get('use_attention', True)
        )
        
        # GNN branch
        gnn_embedding_dim = gnn_config.get('embedding_dim', 128)
        self.gnn = ConnectivityGNN(
            node_feature_dim=gnn_config.get('node_feature_dim', 5),
            hidden_dim=gnn_config.get('hidden_dim', 64),
            n_layers=gnn_config.get('n_layers', 3),
            n_classes=n_classes,
            embedding_dim=gnn_embedding_dim,
            dropout=gnn_config.get('dropout', 0.3),
            pool_type=gnn_config.get('pool_type', 'attention'),
            gnn_type=gnn_config.get('gnn_type', 'gat')
        )
        
        # Fusion module
        if fusion_type == 'attention':
            self.fusion = FusionAttention(
                cnn_embedding_dim,
                gnn_embedding_dim,
                hidden_dim=fc_hidden_dims[0]
            )
            fused_dim = fc_hidden_dims[0]
        elif fusion_type == 'concat':
            self.fusion = None
            fused_dim = cnn_embedding_dim + gnn_embedding_dim
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(cnn_embedding_dim, gnn_embedding_dim)
            fused_dim = max(cnn_embedding_dim, gnn_embedding_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Final classifier
        fc_layers = []
        in_dim = fused_dim
        for hidden_dim in fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        fc_layers.append(nn.Linear(in_dim, n_classes))
        self.classifier = nn.Sequential(*fc_layers)
        
        # Store modality weights for interpretation
        self.modality_weights = None
        
    def forward(
        self,
        spectrogram: torch.Tensor,
        graph: Union[Data, Batch],
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Args:
            spectrogram: CNN input, shape (batch, channels, freq, time)
            graph: PyG Data/Batch for GNN
            return_features: If True, return intermediate features
            
        Returns:
            logits: Classification logits
            features: Dict of intermediate features (if return_features=True)
        """
        # Extract embeddings from each branch
        cnn_embedding = self.cnn.get_embedding(spectrogram)
        gnn_embedding = self.gnn.get_embedding(graph)
        
        # Fuse embeddings
        if self.fusion_type == 'attention':
            fused, weights = self.fusion(cnn_embedding, gnn_embedding)
            self.modality_weights = weights
        elif self.fusion_type == 'concat':
            fused = torch.cat([cnn_embedding, gnn_embedding], dim=-1)
        elif self.fusion_type == 'gated':
            fused, weights = self.fusion(cnn_embedding, gnn_embedding)
            self.modality_weights = weights
        
        # Classification
        logits = self.classifier(fused)
        
        if return_features:
            features = {
                'cnn_embedding': cnn_embedding,
                'gnn_embedding': gnn_embedding,
                'fused_embedding': fused,
                'modality_weights': self.modality_weights
            }
            return logits, features
        
        return logits
    
    def get_modality_importance(self) -> Optional[torch.Tensor]:
        """Get learned weights for each modality."""
        return self.modality_weights
    
    def forward_cnn_only(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Forward pass using only CNN branch."""
        return self.cnn(spectrogram)
    
    def forward_gnn_only(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass using only GNN branch."""
        return self.gnn(graph)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining modality embeddings.
    
    Uses learned gates to control information flow from each modality.
    """
    
    def __init__(self, dim1: int, dim2: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(dim1, dim2)
        
        self.out_dim = hidden_dim
        
        # Project to common dimension
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        
        # Gate networks
        self.gate1 = nn.Sequential(
            nn.Linear(dim1 + dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.gate2 = nn.Sequential(
            nn.Linear(dim1 + dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: First modality embedding
            x2: Second modality embedding
            
        Returns:
            fused: Gated fusion output
            gate_values: Gate activation values
        """
        # Project to common dimension
        h1 = self.proj1(x1)
        h2 = self.proj2(x2)
        
        # Compute gates
        concat = torch.cat([x1, x2], dim=-1)
        g1 = self.gate1(concat)
        g2 = self.gate2(concat)
        
        # Gated combination
        fused = g1 * h1 + g2 * h2
        
        # Mean gate values for interpretation
        gate_values = torch.stack([g1.mean(dim=-1), g2.mean(dim=-1)], dim=-1)
        
        return fused, gate_values


class EnsembleClassifier(nn.Module):
    """
    Ensemble of hybrid models for uncertainty estimation.
    """
    
    def __init__(
        self,
        n_models: int = 5,
        **model_kwargs
    ):
        super().__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            HybridClassifier(**model_kwargs) for _ in range(n_models)
        ])
        
    def forward(
        self,
        spectrogram: torch.Tensor,
        graph: Union[Data, Batch]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectrogram: CNN input
            graph: GNN input
            
        Returns:
            mean_logits: Average logits across ensemble
            std_logits: Standard deviation of logits (uncertainty)
        """
        all_logits = []
        
        for model in self.models:
            logits = model(spectrogram, graph)
            all_logits.append(logits)
        
        all_logits = torch.stack(all_logits, dim=0)  # (n_models, batch, n_classes)
        
        mean_logits = all_logits.mean(dim=0)
        std_logits = all_logits.std(dim=0)
        
        return mean_logits, std_logits
    
    def predict_with_uncertainty(
        self,
        spectrogram: torch.Tensor,
        graph: Union[Data, Batch]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty estimates.
        
        Returns:
            predictions: Predicted class labels
            confidences: Prediction confidence
            uncertainties: Prediction uncertainty
        """
        mean_logits, std_logits = self.forward(spectrogram, graph)
        
        probs = F.softmax(mean_logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1)[0]
        uncertainties = std_logits.mean(dim=-1)
        
        return predictions, confidences, uncertainties


def create_hybrid_model(
    n_classes: int = 3,
    cnn_config: Optional[Dict] = None,
    gnn_config: Optional[Dict] = None,
    fusion_type: str = 'attention',
    ensemble_size: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create hybrid model.
    
    Args:
        n_classes: Number of classes
        cnn_config: CNN configuration
        gnn_config: GNN configuration
        fusion_type: Fusion method
        ensemble_size: If > 1, creates ensemble
        
    Returns:
        Hybrid model or ensemble
    """
    if ensemble_size > 1:
        return EnsembleClassifier(
            n_models=ensemble_size,
            n_classes=n_classes,
            cnn_config=cnn_config,
            gnn_config=gnn_config,
            fusion_type=fusion_type,
            **kwargs
        )
    
    return HybridClassifier(
        n_classes=n_classes,
        cnn_config=cnn_config,
        gnn_config=gnn_config,
        fusion_type=fusion_type,
        **kwargs
    )
