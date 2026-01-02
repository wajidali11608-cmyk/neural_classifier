"""
GNN Branch for Brain Connectivity Graph Classification

Graph Neural Network for analyzing functional connectivity patterns
in brain networks represented as graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.data import Data, Batch

from .attention import GraphAttentionPooling


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer with residual connection."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        residual: bool = True
    ):
        super().__init__()
        
        self.conv = GCNConv(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        self.residual = residual
        if residual and in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features)
        else:
            self.res_proj = None
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        identity = x
        
        x = self.conv(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            x = x + identity
            
        return x


class GATLayer(nn.Module):
    """Graph Attention Network layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        
        self.conv = GATConv(
            in_features, out_features,
            heads=heads,
            dropout=dropout,
            concat=concat
        )
        
        actual_out = out_features * heads if concat else out_features
        self.bn = nn.BatchNorm1d(actual_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attn_weights = self.conv(
            x, edge_index,
            edge_attr=edge_attr.squeeze(-1) if edge_attr is not None else None,
            return_attention_weights=True
        )
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        return x, attn_weights


class ConnectivityGNN(nn.Module):
    """
    Graph Neural Network for brain connectivity classification.
    
    Analyzes connectivity graphs where:
    - Nodes represent brain regions/channels
    - Edges represent functional connectivity (coherence)
    - Node features are spectral properties (band powers)
    
    Args:
        node_feature_dim: Dimension of node features
        hidden_dim: Hidden layer dimension
        n_layers: Number of GNN layers
        n_classes: Number of output classes
        embedding_dim: Dimension of graph embedding
        dropout: Dropout probability
        pool_type: Global pooling type ('mean', 'max', 'attention')
        gnn_type: GNN layer type ('gcn', 'gat', 'sage')
        heads: Number of attention heads (for GAT)
    """
    
    def __init__(
        self,
        node_feature_dim: int = 5,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_classes: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        pool_type: str = 'attention',
        gnn_type: str = 'gat',
        heads: int = 4
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        current_dim = hidden_dim
        
        for i in range(n_layers):
            if gnn_type == 'gcn':
                layer = GCNLayer(current_dim, hidden_dim, dropout)
                current_dim = hidden_dim
            elif gnn_type == 'gat':
                concat = i < n_layers - 1  # Don't concat on last layer
                layer = GATLayer(current_dim, hidden_dim, heads, dropout, concat)
                current_dim = hidden_dim * heads if concat else hidden_dim
            elif gnn_type == 'sage':
                layer = SAGEConv(current_dim, hidden_dim)
                current_dim = hidden_dim
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
                
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(current_dim))
        
        # Global pooling
        self.pool_type = pool_type
        if pool_type == 'attention':
            self.pool = GraphAttentionPooling(current_dim, hidden_dim)
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(current_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
        # For storing attention weights
        self.attention_weights = None
        
    def forward(
        self,
        data: Union[Data, Batch],
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Args:
            data: PyG Data or Batch object with x, edge_index, edge_attr, batch
            return_embedding: If True, return graph embedding
            
        Returns:
            logits or embedding
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'gat':
                x, attn = layer(x, edge_index, edge_attr)
                if i == len(self.gnn_layers) - 1:
                    self.attention_weights = attn
            elif self.gnn_type == 'gcn':
                edge_weight = edge_attr.squeeze(-1) if edge_attr is not None else None
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x, edge_index)
                x = F.relu(x)
            
            x = self.layer_norms[i](x)
        
        # Global pooling
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            x = global_max_pool(x, batch)
        elif self.pool_type == 'attention':
            x, pool_attn = self.pool(x, batch)
            self.attention_weights = pool_attn
        
        # FC -> embedding
        embedding = self.fc(x)
        
        if return_embedding:
            return embedding
        
        # Classification
        logits = self.classifier(embedding)
        
        return logits
    
    def get_embedding(self, data: Union[Data, Batch]) -> torch.Tensor:
        """Extract graph embedding without classification."""
        return self.forward(data, return_embedding=True)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return stored attention weights."""
        return self.attention_weights


class EdgeWeightedGNN(nn.Module):
    """
    GNN with learnable edge weighting.
    
    Learns to weight edges based on connectivity importance.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 5,
        edge_feature_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_classes: int = 3
    ):
        super().__init__()
        
        # Edge weight predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim + 2 * node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Node encoder
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Compute edge weights
        row, col = edge_index
        edge_features = torch.cat([
            x[row], x[col],
            edge_attr
        ], dim=-1)
        edge_weights = self.edge_mlp(edge_features).squeeze(-1)
        
        # Encode nodes
        x = self.node_encoder(x)
        
        # GNN forward
        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)
        
        # Pool and classify
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def create_gnn_model(
    node_feature_dim: int = 5,
    n_classes: int = 3,
    model_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create GNN model.
    
    Args:
        node_feature_dim: Node feature dimension
        n_classes: Number of classes
        model_type: 'standard' or 'edge_weighted'
        **kwargs: Additional arguments
        
    Returns:
        GNN model
    """
    if model_type == 'standard':
        return ConnectivityGNN(node_feature_dim, n_classes=n_classes, **kwargs)
    elif model_type == 'edge_weighted':
        return EdgeWeightedGNN(node_feature_dim, n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
