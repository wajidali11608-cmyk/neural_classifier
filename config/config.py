"""
Neural Activity Classification Pipeline Configuration

Defines all hyperparameters, model settings, and feature extraction parameters
for classifying neural signals into healthy, schizophrenia, and bipolar disorder.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    sample_rate: float = 256.0  # Hz
    n_channels: int = 64  # Number of EEG/recording channels
    window_size: float = 4.0  # seconds
    window_overlap: float = 0.5  # 50% overlap
    
    # Class labels
    class_names: List[str] = field(default_factory=lambda: [
        'healthy', 'schizophrenia', 'bipolar'
    ])
    n_classes: int = 3
    
    # Filtering
    lowcut: float = 0.5  # Hz
    highcut: float = 100.0  # Hz
    notch_freq: float = 50.0  # Hz (powerline interference)
    filter_order: int = 4


@dataclass
class FrequencyBands:
    """EEG frequency band definitions."""
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 100.0)
    
    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return bands as dictionary."""
        return {
            'delta': self.delta,
            'theta': self.theta,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }
    
    def get_band_names(self) -> List[str]:
        """Return list of band names."""
        return ['delta', 'theta', 'alpha', 'beta', 'gamma']


@dataclass
class SpectralConfig:
    """Spectral feature extraction configuration."""
    n_fft: int = 256
    hop_length: int = 64
    n_mels: int = 64
    fmin: float = 0.5
    fmax: float = 100.0
    
    # PSD computation
    psd_method: str = 'welch'
    psd_nperseg: int = 256
    psd_noverlap: int = 128


@dataclass
class GraphConfig:
    """Graph construction configuration."""
    coherence_threshold: float = 0.3  # Minimum coherence for edge
    use_weighted_edges: bool = True
    max_edges_per_node: int = 10
    normalize_edge_weights: bool = True


@dataclass
class CNNConfig:
    """CNN branch architecture configuration."""
    in_channels: int = 1
    base_filters: int = 32
    n_conv_blocks: int = 4
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    dropout_rate: float = 0.5
    fc_hidden_dim: int = 256
    embedding_dim: int = 128
    use_batch_norm: bool = True
    use_attention: bool = True


@dataclass
class GNNConfig:
    """GNN branch architecture configuration."""
    node_feature_dim: int = 5  # Band powers as node features
    hidden_dim: int = 64
    n_layers: int = 3
    heads: int = 4  # For GAT
    dropout_rate: float = 0.3
    embedding_dim: int = 128
    pool_type: str = 'mean'  # 'mean', 'max', 'attention'
    use_edge_features: bool = True
    gnn_type: str = 'gat'  # 'gcn', 'gat', 'graphsage'


@dataclass
class HybridConfig:
    """Hybrid fusion model configuration."""
    fusion_type: str = 'attention'  # 'concat', 'attention', 'gated'
    fc_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.5
    use_residual: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs: int = 100
    patience: int = 15  # Early stopping patience
    
    # K-fold cross-validation
    n_folds: int = 5
    
    # Learning rate scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 5
    
    # Optimizer
    optimizer: str = 'adamw'
    
    # Loss
    use_class_weights: bool = True
    label_smoothing: float = 0.1


@dataclass
class UncertaintyConfig:
    """Uncertainty estimation configuration."""
    mc_dropout_samples: int = 30
    ensemble_size: int = 5
    calibration_method: str = 'temperature_scaling'


@dataclass
class ExplainabilityConfig:
    """Explainability configuration."""
    grad_cam_layer: str = 'conv_blocks.3'  # Target layer for Grad-CAM
    n_top_edges: int = 20  # Top connectivity edges to highlight
    saliency_smooth_sigma: float = 1.0


@dataclass 
class Config:
    """Master configuration combining all settings."""
    data: DataConfig = field(default_factory=DataConfig)
    bands: FrequencyBands = field(default_factory=FrequencyBands)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    
    # Device
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Output paths
    # Output paths
    output_dir: str = 'outputs'
    model_save_dir: str = 'outputs/models'
    plot_save_dir: str = 'outputs/plots'

    def __post_init__(self):
        # Create timestamped subdirectory
        import datetime
        from pathlib import Path
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = Path(self.output_dir) / f"run_{timestamp}"
        
        self.output_dir = str(base_dir)
        self.model_save_dir = str(base_dir / 'models')
        self.plot_save_dir = str(base_dir / 'plots')


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def get_band_limits(config: Config) -> Dict[str, Tuple[float, float]]:
    """Extract frequency band limits from config."""
    return config.bands.as_dict()
