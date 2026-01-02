"""
Graph Builder for Brain Connectivity Networks

Constructs PyTorch Geometric graph representations from coherence
matrices for GNN-based analysis of brain connectivity.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch

from .spectral import extract_band_power, get_frequency_bands
from .coherence import compute_coherence_matrix, threshold_connectivity


def create_adjacency_matrix(
    coherence_matrix: np.ndarray,
    threshold: float = 0.3,
    weighted: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create adjacency matrix and edge weights from coherence.
    
    Args:
        coherence_matrix: Coherence between channels, shape (n_channels, n_channels)
        threshold: Minimum coherence for edge creation
        weighted: If True, return edge weights
        
    Returns:
        edge_index: Shape (2, n_edges) - source and target indices
        edge_weights: Shape (n_edges,) - coherence values as weights
    """
    # Threshold matrix
    adj = threshold_connectivity(coherence_matrix, threshold, method='absolute')
    
    # Get edge indices (non-zero entries)
    rows, cols = np.where(adj > 0)
    
    edge_index = np.stack([rows, cols], axis=0)
    edge_weights = adj[rows, cols] if weighted else np.ones(len(rows))
    
    return edge_index, edge_weights


def compute_node_features(
    data: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Compute node features for each channel.
    
    Uses band power as primary features for each channel/node.
    
    Args:
        data: Signal data, shape (n_channels, n_timepoints)
        fs: Sampling frequency
        bands: Frequency bands (uses default EEG bands if None)
        
    Returns:
        node_features: Shape (n_channels, n_features)
    """
    if bands is None:
        bands = get_frequency_bands()
    
    # Extract band power per channel
    band_powers = extract_band_power(data, fs, bands)
    
    # Stack features: [delta, theta, alpha, beta, gamma]
    feature_list = []
    for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        if band_name in band_powers:
            feature_list.append(band_powers[band_name])
    
    node_features = np.stack(feature_list, axis=1)
    
    # Normalize features
    mean = np.mean(node_features, axis=0, keepdims=True)
    std = np.std(node_features, axis=0, keepdims=True) + 1e-8
    node_features = (node_features - mean) / std
    
    return node_features


def build_connectivity_graph(
    data: np.ndarray,
    fs: float,
    coherence_threshold: float = 0.3,
    nperseg: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    device: str = 'cpu'
) -> Data:
    """
    Build PyTorch Geometric graph from neural signal.
    
    Creates a graph where:
    - Nodes = channels with band power features
    - Edges = coherence-based connectivity
    
    Args:
        data: Multi-channel signal, shape (n_channels, n_timepoints)
        fs: Sampling frequency
        coherence_threshold: Minimum coherence for edge
        nperseg: Segment length for coherence computation
        bands: Frequency bands
        device: Device for tensors
        
    Returns:
        PyG Data object with node features, edge index, and edge weights
    """
    if bands is None:
        bands = get_frequency_bands()
    
    n_channels = data.shape[0]
    
    # Compute node features (band powers)
    node_features = compute_node_features(data, fs, bands)
    
    # Compute coherence matrix (average across frequencies)
    coh_matrix, _ = compute_coherence_matrix(data, fs, nperseg)
    mean_coh = np.mean(coh_matrix, axis=2)
    
    # Create adjacency
    edge_index, edge_weights = create_adjacency_matrix(
        mean_coh, coherence_threshold, weighted=True
    )
    
    # Convert to tensors
    x = torch.FloatTensor(node_features).to(device)
    edge_index = torch.LongTensor(edge_index).to(device)
    edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1).to(device)
    
    # Create PyG Data object
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_channels
    )
    
    return graph


def build_batch_graphs(
    signals: np.ndarray,
    fs: float,
    coherence_threshold: float = 0.3,
    nperseg: int = 256,
    device: str = 'cpu'
) -> Batch:
    """
    Build batch of graphs from multiple samples.
    
    Args:
        signals: Shape (n_samples, n_channels, n_timepoints)
        fs: Sampling frequency
        coherence_threshold: Edge threshold
        nperseg: Segment length
        device: Device
        
    Returns:
        PyG Batch of graphs
    """
    graphs = []
    
    for i in range(signals.shape[0]):
        graph = build_connectivity_graph(
            signals[i], fs, coherence_threshold, nperseg, device=device
        )
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


def compute_graph_statistics(graph: Data) -> Dict[str, float]:
    """
    Compute statistics of a connectivity graph.
    
    Args:
        graph: PyG Data object
        
    Returns:
        Dictionary of graph metrics
    """
    n_nodes = graph.num_nodes
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    
    stats = {
        'num_nodes': n_nodes,
        'num_edges': n_edges,
        'edge_density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    }
    
    if graph.edge_attr is not None:
        edge_weights = graph.edge_attr.cpu().numpy().flatten()
        stats['mean_edge_weight'] = float(np.mean(edge_weights))
        stats['max_edge_weight'] = float(np.max(edge_weights))
        stats['std_edge_weight'] = float(np.std(edge_weights))
    
    # Node degree
    if n_edges > 0:
        edge_index_np = graph.edge_index.cpu().numpy()
        degrees = np.bincount(edge_index_np[0], minlength=n_nodes)
        stats['mean_degree'] = float(np.mean(degrees))
        stats['max_degree'] = float(np.max(degrees))
    
    return stats


def augment_graph(
    graph: Data,
    drop_edge_prob: float = 0.1,
    drop_node_prob: float = 0.05,
    noise_std: float = 0.1
) -> Data:
    """
    Apply data augmentation to graph.
    
    Args:
        graph: Input graph
        drop_edge_prob: Probability of dropping each edge
        drop_node_prob: Probability of masking each node
        noise_std: Standard deviation of added noise to features
        
    Returns:
        Augmented graph
    """
    # Clone graph
    x = graph.x.clone()
    edge_index = graph.edge_index.clone()
    edge_attr = graph.edge_attr.clone() if graph.edge_attr is not None else None
    
    # Add noise to node features
    x = x + torch.randn_like(x) * noise_std
    
    # Drop edges randomly
    if drop_edge_prob > 0 and edge_index.shape[1] > 0:
        mask = torch.rand(edge_index.shape[1]) > drop_edge_prob
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
    
    # Mask node features randomly
    if drop_node_prob > 0:
        node_mask = torch.rand(x.shape[0]) > drop_node_prob
        x = x * node_mask.float().unsqueeze(1)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=graph.num_nodes
    )


def create_fully_connected_graph(
    data: np.ndarray,
    fs: float,
    device: str = 'cpu'
) -> Data:
    """
    Create fully connected graph with coherence as edge weights.
    
    Useful when you want to learn the graph structure.
    
    Args:
        data: Signal data
        fs: Sampling frequency
        device: Device
        
    Returns:
        Fully connected graph
    """
    n_channels = data.shape[0]
    
    # Node features
    node_features = compute_node_features(data, fs)
    
    # Full coherence matrix
    coh_matrix, _ = compute_coherence_matrix(data, fs)
    mean_coh = np.mean(coh_matrix, axis=2)
    
    # Create full adjacency (excluding self-loops)
    rows, cols = np.where(~np.eye(n_channels, dtype=bool))
    edge_index = np.stack([rows, cols], axis=0)
    edge_weights = mean_coh[rows, cols]
    
    # Convert to tensors
    x = torch.FloatTensor(node_features).to(device)
    edge_index = torch.LongTensor(edge_index).to(device)
    edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1).to(device)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_channels
    )
