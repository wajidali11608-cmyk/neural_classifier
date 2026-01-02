"""Feature engineering module for neural signal analysis."""
from .spectral import (
    compute_stft, compute_psd, extract_band_power,
    compute_spectrogram_features, get_frequency_bands
)
from .coherence import (
    compute_coherence, compute_coherence_matrix,
    compute_connectivity_features
)
from .graph_builder import (
    build_connectivity_graph, build_batch_graphs,
    create_adjacency_matrix
)

__all__ = [
    'compute_stft', 'compute_psd', 'extract_band_power',
    'compute_spectrogram_features', 'get_frequency_bands',
    'compute_coherence', 'compute_coherence_matrix',
    'compute_connectivity_features',
    'build_connectivity_graph', 'build_batch_graphs',
    'create_adjacency_matrix'
]
