"""
GNN Explainability for Connectivity Analysis

Extracts attention weights, identifies important edges/connections,
and provides interpretable connectivity biomarkers.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data


def extract_attention_weights(
    model: nn.Module,
    graph: Data,
    layer_idx: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract attention weights from GNN.
    
    Works with GAT-based models that compute attention.
    
    Args:
        model: GNN model
        graph: Input graph
        layer_idx: Index of layer to extract from (-1 = last)
        
    Returns:
        edge_index: Edge indices
        attention_weights: Attention weights per edge
    """
    model.eval()
    
    with torch.no_grad():
        _ = model(graph)
    
    # Get stored attention weights
    attn_weights = model.get_attention_weights()
    
    if attn_weights is None:
        # No attention available, return uniform weights
        n_edges = graph.edge_index.shape[1]
        return graph.edge_index.cpu().numpy(), np.ones(n_edges) / n_edges
    
    if isinstance(attn_weights, tuple):
        edge_index, weights = attn_weights
        edge_index = edge_index.cpu().numpy()
        weights = weights.cpu().numpy()
        
        # Average over attention heads if multi-head
        if weights.ndim > 1:
            weights = weights.mean(axis=-1)
    else:
        # Pool attention is a vector of node importance
        edge_index = graph.edge_index.cpu().numpy()
        weights = attn_weights.cpu().numpy()
    
    return edge_index, weights


def identify_important_edges(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    coherence_values: Optional[np.ndarray] = None,
    top_k: int = 20,
    channel_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Identify most important connectivity edges.
    
    Args:
        edge_index: Edge source/target indices, shape (2, n_edges)
        edge_weights: Importance weights per edge
        coherence_values: Original coherence values
        top_k: Number of top edges to return
        channel_names: Names of channels/nodes
        
    Returns:
        List of dictionaries with edge information
    """
    n_edges = edge_index.shape[1]
    
    # Sort by importance
    importance_order = np.argsort(edge_weights)[::-1][:top_k]
    
    important_edges = []
    
    for idx in importance_order:
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        
        edge_info = {
            'source': src,
            'target': dst,
            'importance': float(edge_weights[idx]),
        }
        
        if channel_names is not None:
            edge_info['source_name'] = channel_names[src]
            edge_info['target_name'] = channel_names[dst]
        
        if coherence_values is not None:
            edge_info['coherence'] = float(coherence_values[idx])
        
        important_edges.append(edge_info)
    
    return important_edges


def extract_biomarkers(
    model: nn.Module,
    graph: Data,
    frequencies: Optional[np.ndarray] = None
) -> Dict:
    """
    Extract interpretable biomarkers from GNN analysis.
    
    Provides connectivity biomarkers for clinical interpretation.
    
    Args:
        model: GNN model
        graph: Input graph
        frequencies: Frequency axis for spectral features
        
    Returns:
        Dictionary of biomarkers
    """
    model.eval()
    
    biomarkers = {}
    
    # Node features are band powers
    node_features = graph.x.cpu().numpy()
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Mean power per band across nodes
    for i, band in enumerate(band_names):
        if i < node_features.shape[1]:
            biomarkers[f'mean_{band}_power'] = float(node_features[:, i].mean())
            biomarkers[f'std_{band}_power'] = float(node_features[:, i].std())
    
    # Node degree (connectivity pattern)
    edge_index = graph.edge_index.cpu().numpy()
    degrees = np.bincount(edge_index[0], minlength=graph.num_nodes)
    
    biomarkers['mean_degree'] = float(degrees.mean())
    biomarkers['max_degree'] = int(degrees.max())
    biomarkers['degree_variance'] = float(degrees.var())
    
    # Identify hub nodes (high degree)
    hub_threshold = np.percentile(degrees, 90)
    hub_nodes = np.where(degrees >= hub_threshold)[0].tolist()
    biomarkers['hub_nodes'] = hub_nodes
    biomarkers['n_hubs'] = len(hub_nodes)
    
    # Edge weights (coherence)
    if graph.edge_attr is not None:
        edge_weights = graph.edge_attr.cpu().numpy().flatten()
        biomarkers['mean_coherence'] = float(edge_weights.mean())
        biomarkers['max_coherence'] = float(edge_weights.max())
        biomarkers['coherence_variance'] = float(edge_weights.var())
    
    # Extract attention-based importance if available
    try:
        edge_index, attn_weights = extract_attention_weights(model, graph)
        important_edges = identify_important_edges(edge_index, attn_weights, top_k=10)
        biomarkers['top_connections'] = important_edges
    except Exception:
        pass
    
    return biomarkers


def visualize_edge_importance(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    n_nodes: int,
    normalize: bool = True
) -> np.ndarray:
    """
    Create adjacency matrix visualization of edge importance.
    
    Args:
        edge_index: Edge indices
        edge_weights: Edge importance weights
        n_nodes: Number of nodes
        normalize: Normalize weights to [0, 1]
        
    Returns:
        Importance matrix, shape (n_nodes, n_nodes)
    """
    importance_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        importance_matrix[src, dst] = edge_weights[i]
    
    # Symmetrize
    importance_matrix = (importance_matrix + importance_matrix.T) / 2
    
    if normalize:
        max_val = importance_matrix.max()
        if max_val > 0:
            importance_matrix /= max_val
    
    return importance_matrix


def compare_class_connectivity(
    model: nn.Module,
    healthy_graphs: List[Data],
    disease_graphs: List[Data],
    top_k: int = 10
) -> Dict:
    """
    Compare connectivity patterns between classes.
    
    Identifies edges that differentiate healthy from disease.
    
    Args:
        model: GNN model
        healthy_graphs: List of graphs from healthy subjects
        disease_graphs: List of graphs from disease subjects
        top_k: Number of differentiating edges to return
        
    Returns:
        Dictionary with comparison results
    """
    def get_mean_importance(graphs):
        all_weights = []
        for graph in graphs:
            _, weights = extract_attention_weights(model, graph)
            all_weights.append(weights)
        return np.mean(all_weights, axis=0)
    
    healthy_importance = get_mean_importance(healthy_graphs)
    disease_importance = get_mean_importance(disease_graphs)
    
    # Difference in importance
    diff = disease_importance - healthy_importance
    
    # Find edges more important in disease
    disease_elevated = np.argsort(diff)[-top_k:]
    
    # Find edges more important in healthy
    healthy_elevated = np.argsort(diff)[:top_k]
    
    return {
        'disease_elevated_edges': disease_elevated.tolist(),
        'healthy_elevated_edges': healthy_elevated.tolist(),
        'importance_difference': diff.tolist()
    }


class NodeImportance:
    """
    Compute node (channel) importance through gradient analysis.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def __call__(
        self,
        graph: Data,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute importance of each node.
        
        Args:
            graph: Input graph
            target_class: Class to explain
            
        Returns:
            Node importance scores
        """
        self.model.train()  # Enable gradients
        
        # Make features require gradients
        graph.x = graph.x.clone()
        graph.x.requires_grad = True
        
        # Forward pass
        output = self.model(graph)
        
        if target_class is None:
            target_class = output.argmax(dim=-1).item()
        
        # Backward pass
        self.model.zero_grad() 
        output[0, target_class].backward()
        
        # Node importance = gradient magnitude
        importance = graph.x.grad.abs().sum(dim=-1)
        
        self.model.eval()
        
        return importance.cpu().numpy()


def get_channel_rankings(
    node_importance: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Rank channels by importance.
    
    Args:
        node_importance: Importance scores
        channel_names: Optional channel names
        
    Returns:
        List of channels ranked by importance
    """
    rankings = []
    
    sorted_indices = np.argsort(node_importance)[::-1]
    
    for rank, idx in enumerate(sorted_indices):
        channel_info = {
            'rank': rank + 1,
            'channel_idx': int(idx),
            'importance': float(node_importance[idx])
        }
        
        if channel_names is not None:
            channel_info['channel_name'] = channel_names[idx]
        
        rankings.append(channel_info)
    
    return rankings
