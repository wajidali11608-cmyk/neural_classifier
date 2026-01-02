"""
Visualization Functions for Neural Classification

Comprehensive plotting utilities for spectrograms, confusion matrices,
saliency maps, connectivity graphs, calibration, and ROC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import networkx as nx


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_spectrogram(
    spectrogram: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    title: str = 'Spectrogram',
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot a spectrogram heatmap.
    
    Args:
        spectrogram: 2D array, shape (n_freq, n_time)
        frequencies: Frequency axis values
        times: Time axis values
        title: Plot title
        cmap: Colormap
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if frequencies is None:
        frequencies = np.arange(spectrogram.shape[0])
    if times is None:
        times = np.arange(spectrogram.shape[1])
    
    im = ax.pcolormesh(
        times, frequencies, spectrogram,
        shading='auto', cmap=cmap
    )
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as annotated heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        normalize: Show percentages
        title: Plot title
        cmap: Colormap
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_display = cm
        fmt = 'd'
        vmax = cm.max()
    
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va="bottom")
    
    # Labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_display[i, j]
            text_color = "white" if value > thresh else "black"
            
            if normalize:
                text = f'{value:.2f}\n({cm[i, j]})'
            else:
                text = str(cm[i, j])
            
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_saliency_map(
    spectrogram: np.ndarray,
    saliency: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    title: str = 'Saliency Map Overlay',
    alpha: float = 0.6,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot saliency map overlaid on spectrogram.
    
    Args:
        spectrogram: Original spectrogram
        saliency: Saliency/Grad-CAM map
        frequencies: Frequency axis
        times: Time axis
        title: Plot title
        alpha: Overlay transparency
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    if frequencies is None:
        frequencies = np.arange(spectrogram.shape[0])
    if times is None:
        times = np.arange(spectrogram.shape[1])
    
    # Original spectrogram
    axes[0].pcolormesh(times, frequencies, spectrogram, shading='auto', cmap='viridis')
    axes[0].set_title('Original Spectrogram', fontsize=12)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    
    # Saliency map
    axes[1].pcolormesh(times, frequencies, saliency, shading='auto', cmap='hot')
    axes[1].set_title('Saliency Map', fontsize=12)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    
    # Overlay
    axes[2].pcolormesh(times, frequencies, spectrogram, shading='auto', cmap='viridis')
    axes[2].pcolormesh(times, frequencies, saliency, shading='auto', cmap='Reds', alpha=alpha)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_connectivity_graph(
    adjacency_matrix: np.ndarray,
    node_importance: Optional[np.ndarray] = None,
    edge_threshold: float = 0.3,
    node_labels: Optional[List[str]] = None,
    title: str = 'Brain Connectivity Graph',
    layout: str = 'spring',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize brain connectivity as a network graph.
    
    Args:
        adjacency_matrix: Connectivity matrix, shape (n_nodes, n_nodes)
        node_importance: Optional node importance for sizing
        edge_threshold: Minimum weight to show edge
        node_labels: Labels for nodes
        title: Plot title
        layout: Graph layout ('spring', 'circular', 'spectral')
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create networkx graph
    G = nx.Graph()
    n_nodes = adjacency_matrix.shape[0]
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add edges with weights
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = adjacency_matrix[i, j]
            if weight > edge_threshold:
                G.add_edge(i, j, weight=weight)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Node sizes
    if node_importance is not None:
        node_sizes = 300 + 700 * (node_importance / node_importance.max())
    else:
        node_sizes = 500
    
    # Edge widths
    edges = G.edges(data=True)
    weights = [e[2]['weight'] for e in edges]
    if weights:
        max_weight = max(weights)
        edge_widths = [1 + 4 * (w / max_weight) for w in weights]
    else:
        edge_widths = []
    
    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color='lightblue',
        edgecolors='navy',
        linewidths=2
    )
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.6,
        edge_color=weights,
        edge_cmap=plt.cm.YlOrRd
    )
    
    # Labels
    if node_labels is None:
        node_labels = {i: str(i) for i in range(n_nodes)}
    else:
        node_labels = {i: node_labels[i] for i in range(len(node_labels))}
    
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=edge_threshold, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Connectivity Strength', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = 'Calibration Curve',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Compute calibration
    confidences = np.max(y_prob, axis=-1) if y_prob.ndim > 1 else y_prob
    predictions = np.argmax(y_prob, axis=-1) if y_prob.ndim > 1 else (y_prob > 0.5).astype(int)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = accuracies[in_bin].mean()
            bin_confidences[i] = confidences[in_bin].mean()
    
    # Main calibration plot
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label='Model')
    
    # Compute ECE
    ece = np.sum(bin_counts / len(y_true) * np.abs(bin_accuracies - bin_confidences))
    
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{title}\nECE: {ece:.4f}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of confidences
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color='gray')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reliability_diagram(
    bin_centers: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: np.ndarray,
    title: str = 'Reliability Diagram',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot reliability diagram from precomputed bins.
    
    Args:
        bin_centers: Center of each bin
        bin_accuracies: Accuracy in each bin
        bin_counts: Count in each bin
        title: Plot title
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    
    # Gap area
    for i, (center, acc) in enumerate(zip(bin_centers, bin_accuracies)):
        if bin_counts[i] > 0:
            gap = abs(acc - center)
            color = 'red' if acc < center else 'green'
            ax.bar(center, gap, bottom=min(acc, center), 
                   width=0.08, alpha=0.3, color=color)
    
    # Accuracy bars
    ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
           edgecolor='black', label='Accuracy')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curves(
    roc_data: Dict[int, Dict],
    class_names: Optional[List[str]] = None,
    title: str = 'ROC Curves (One-vs-Rest)',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot multi-class ROC curves.
    
    Args:
        roc_data: Dictionary with fpr, tpr, auc per class
        class_names: Names of classes
        title: Plot title
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    
    for i, (class_idx, data) in enumerate(roc_data.items()):
        label = class_names[class_idx] if class_names else f'Class {class_idx}'
        auc = data.get('auc', 0)
        
        ax.plot(
            data['fpr'], data['tpr'],
            color=colors[i],
            linewidth=2,
            label=f'{label} (AUC = {auc:.3f})'
        )
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List],
    title: str = 'Training History',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with train/val loss, accuracy, etc.
        title: Plot title
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1
    if 'val_f1' in history:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_band_importance(
    band_importance: Dict[str, float],
    title: str = 'Frequency Band Importance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot importance of frequency bands.
    
    Args:
        band_importance: Dictionary mapping band names to importance values
        title: Plot title
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bands = list(band_importance.keys())
    values = list(band_importance.values())
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(bands)))
    
    bars = ax.bar(bands, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_k: int = 15,
    title: str = 'Feature Importance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot top-k feature importances as horizontal bar chart.
    
    Args:
        feature_names: Names of features
        importance_values: Importance scores
        top_k: Number of top features to show
        title: Plot title
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by importance
    sorted_idx = np.argsort(importance_values)[::-1][:top_k]
    
    names = [feature_names[i] for i in sorted_idx]
    values = importance_values[sorted_idx]
    
    # Horizontal bar chart
    y_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    
    ax.barh(y_pos, values, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Top feature at top
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
