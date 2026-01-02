"""
Coherence and Connectivity Feature Extraction

Computes coherence between channel pairs to measure functional connectivity
in neural signals. Used for constructing brain connectivity graphs.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, List
from itertools import combinations


def compute_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude-squared coherence between two signals.
    
    Coherence measures the linear correlation between two signals
    as a function of frequency.
    
    Args:
        x: First signal, shape (n_timepoints,)
        y: Second signal, shape (n_timepoints,)
        fs: Sampling frequency
        nperseg: Segment length for spectral estimation
        noverlap: Overlap between segments
        
    Returns:
        frequencies: Frequency bins
        coherence: Coherence values [0, 1]
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    f, Cxy = signal.coherence(
        x, y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap
    )
    
    return f, Cxy


def compute_coherence_matrix(
    data: np.ndarray,
    fs: float,
    nperseg: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute coherence matrix between all channel pairs.
    
    Args:
        data: Multi-channel signal, shape (n_channels, n_timepoints)
        fs: Sampling frequency
        nperseg: Segment length
        bands: Optional frequency bands for band-specific coherence
        
    Returns:
        full_coherence: Shape (n_channels, n_channels, n_freq)
        band_coherence: Dict mapping band names to (n_channels, n_channels) matrices
    """
    n_channels = data.shape[0]
    
    # Compute coherence for first pair to get frequency resolution
    f, _ = compute_coherence(data[0], data[1], fs, nperseg)
    n_freq = len(f)
    
    # Initialize coherence matrix
    coherence_matrix = np.zeros((n_channels, n_channels, n_freq))
    
    # Diagonal is 1 (self-coherence)
    for i in range(n_channels):
        coherence_matrix[i, i, :] = 1.0
    
    # Compute coherence for all unique pairs
    for i, j in combinations(range(n_channels), 2):
        _, coh = compute_coherence(data[i], data[j], fs, nperseg)
        coherence_matrix[i, j, :] = coh
        coherence_matrix[j, i, :] = coh  # Symmetric
    
    # Compute band-averaged coherence if bands specified
    band_coherence = {}
    if bands is not None:
        for band_name, (low, high) in bands.items():
            band_mask = (f >= low) & (f <= high)
            if np.any(band_mask):
                band_coh = np.mean(coherence_matrix[:, :, band_mask], axis=2)
                band_coherence[band_name] = band_coh
            else:
                band_coherence[band_name] = np.zeros((n_channels, n_channels))
    
    return coherence_matrix, band_coherence


def compute_mean_coherence(
    coherence_matrix: np.ndarray,
    f: Optional[np.ndarray] = None,
    freq_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute mean coherence across all frequencies or a frequency range.
    
    Args:
        coherence_matrix: Shape (n_channels, n_channels, n_freq)
        f: Frequency bins (required if freq_range specified)
        freq_range: (low, high) frequency range
        
    Returns:
        Mean coherence matrix, shape (n_channels, n_channels)
    """
    if freq_range is not None and f is not None:
        low, high = freq_range
        mask = (f >= low) & (f <= high)
        return np.mean(coherence_matrix[:, :, mask], axis=2)
    
    return np.mean(coherence_matrix, axis=2)


def compute_connectivity_features(
    data: np.ndarray,
    fs: float,
    nperseg: int = 256
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive connectivity features from signal.
    
    Args:
        data: Multi-channel signal, shape (n_channels, n_timepoints)
        fs: Sampling frequency
        nperseg: Segment length for coherence
        
    Returns:
        Dictionary of connectivity features
    """
    from .spectral import get_frequency_bands
    
    bands = get_frequency_bands()
    
    # Compute coherence matrix
    full_coh, band_coh = compute_coherence_matrix(data, fs, nperseg, bands)
    
    features = {}
    
    # Mean coherence (global connectivity)
    features['mean_coherence'] = np.mean(compute_mean_coherence(full_coh))
    
    # Band-specific mean coherence
    for band_name, coh_matrix in band_coh.items():
        # Mean (excluding diagonal)
        n = coh_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        features[f'{band_name}_mean_coherence'] = np.mean(coh_matrix[mask])
        features[f'{band_name}_max_coherence'] = np.max(coh_matrix[mask])
        features[f'{band_name}_std_coherence'] = np.std(coh_matrix[mask])
    
    # Graph-theoretic features
    mean_coh = compute_mean_coherence(full_coh)
    
    # Degree (sum of coherence for each channel)
    features['channel_degree'] = np.sum(mean_coh, axis=1) - 1  # Subtract self
    
    # Clustering coefficient approximation
    features['global_efficiency'] = _compute_efficiency(mean_coh)
    
    return features


def compute_phase_locking_value(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    band: Tuple[float, float]
) -> float:
    """
    Compute Phase Locking Value (PLV) between two signals.
    
    PLV measures phase synchronization independent of amplitude.
    
    Args:
        x, y: Input signals
        fs: Sampling frequency
        band: (low, high) frequency band for filtering
        
    Returns:
        PLV value [0, 1]
    """
    from scipy.signal import butter, filtfilt, hilbert
    
    # Bandpass filter
    nyq = 0.5 * fs
    low = band[0] / nyq
    high = band[1] / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = butter(4, [low, high], btype='band')
    
    x_filt = filtfilt(b, a, x)
    y_filt = filtfilt(b, a, y)
    
    # Compute analytic signal
    x_analytic = hilbert(x_filt)
    y_analytic = hilbert(y_filt)
    
    # Extract phase
    phase_x = np.angle(x_analytic)
    phase_y = np.angle(y_analytic)
    
    # Compute PLV
    phase_diff = phase_x - phase_y
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv


def compute_plv_matrix(
    data: np.ndarray,
    fs: float,
    band: Tuple[float, float]
) -> np.ndarray:
    """
    Compute PLV matrix for all channel pairs.
    
    Args:
        data: Shape (n_channels, n_timepoints)
        fs: Sampling frequency
        band: Frequency band
        
    Returns:
        PLV matrix, shape (n_channels, n_channels)
    """
    n_channels = data.shape[0]
    plv_matrix = np.eye(n_channels)
    
    for i, j in combinations(range(n_channels), 2):
        plv = compute_phase_locking_value(data[i], data[j], fs, band)
        plv_matrix[i, j] = plv
        plv_matrix[j, i] = plv
    
    return plv_matrix


def _compute_efficiency(connectivity_matrix: np.ndarray) -> float:
    """
    Compute global efficiency of connectivity matrix.
    
    Higher efficiency indicates more integrated networks.
    """
    n = connectivity_matrix.shape[0]
    
    # Convert coherence to distance (inverse)
    with np.errstate(divide='ignore'):
        distance = 1.0 / (connectivity_matrix + 1e-10)
    np.fill_diagonal(distance, 0)
    
    # Compute shortest paths (approximation using direct connections)
    efficiency = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                efficiency += 1.0 / (distance[i, j] + 1e-10)
    
    efficiency /= (n * (n - 1))
    
    return efficiency


def threshold_connectivity(
    connectivity_matrix: np.ndarray,
    threshold: float = 0.3,
    method: str = 'absolute'
) -> np.ndarray:
    """
    Threshold connectivity matrix to create sparse adjacency.
    
    Args:
        connectivity_matrix: Dense connectivity matrix
        threshold: Threshold value
        method: 'absolute' - keep values > threshold
                'percentile' - keep top percentile
                'proportional' - keep top proportion
                
    Returns:
        Thresholded matrix (values below threshold set to 0)
    """
    result = connectivity_matrix.copy()
    
    # Exclude diagonal
    n = result.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = result[mask]
    
    if method == 'absolute':
        result[result < threshold] = 0
        
    elif method == 'percentile':
        cutoff = np.percentile(values, 100 - threshold)
        result[result < cutoff] = 0
        
    elif method == 'proportional':
        k = int(len(values) * threshold)
        cutoff = np.sort(values)[-k] if k > 0 else np.inf
        result[result < cutoff] = 0
    
    # Ensure diagonal is 0
    np.fill_diagonal(result, 0)
    
    return result
