"""
Normalization and Segmentation Utilities

Implements signal normalization methods and windowing/segmentation
for neural activity data preprocessing.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import stats


def z_score_normalize(
    data: np.ndarray,
    axis: Optional[int] = -1,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Apply z-score normalization (zero mean, unit variance).
    
    Args:
        data: Input data, any shape
        axis: Axis along which to normalize (default: last axis = time)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized data with same shape
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    
    return (data - mean) / (std + eps)


def min_max_normalize(
    data: np.ndarray,
    axis: Optional[int] = -1,
    feature_range: Tuple[float, float] = (0, 1),
    eps: float = 1e-8
) -> np.ndarray:
    """
    Apply min-max normalization to scale data to a range.
    
    Args:
        data: Input data
        axis: Axis along which to normalize
        feature_range: Target range (min, max)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized data scaled to feature_range
    """
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    
    # Scale to [0, 1]
    scaled = (data - min_val) / (max_val - min_val + eps)
    
    # Scale to feature_range
    range_min, range_max = feature_range
    return scaled * (range_max - range_min) + range_min


def robust_normalize(
    data: np.ndarray,
    axis: Optional[int] = -1,
    quantile_range: Tuple[float, float] = (25.0, 75.0)
) -> np.ndarray:
    """
    Apply robust normalization using median and IQR.
    
    More robust to outliers than z-score normalization.
    
    Args:
        data: Input data
        axis: Axis along which to normalize
        quantile_range: Percentiles for computing scale
        
    Returns:
        Normalized data
    """
    median = np.median(data, axis=axis, keepdims=True)
    
    q_low = np.percentile(data, quantile_range[0], axis=axis, keepdims=True)
    q_high = np.percentile(data, quantile_range[1], axis=axis, keepdims=True)
    iqr = q_high - q_low
    
    return (data - median) / (iqr + 1e-8)


def segment_signal(
    data: np.ndarray,
    window_size: int,
    overlap: float = 0.5,
    pad_mode: str = 'reflect'
) -> np.ndarray:
    """
    Segment signal into fixed-length windows.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints)
        window_size: Number of timepoints per window
        overlap: Fraction of overlap between windows (0 to 1)
        pad_mode: Padding mode for edge windows ('reflect', 'constant', 'edge')
        
    Returns:
        Segmented data, shape (n_windows, n_channels, window_size)
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_timepoints = data.shape
    
    # Calculate step size
    step = int(window_size * (1 - overlap))
    step = max(step, 1)  # Ensure at least 1 step
    
    # Calculate number of windows
    n_windows = (n_timepoints - window_size) // step + 1
    
    # Handle edge case where signal is shorter than window
    if n_windows < 1:
        # Pad the signal
        pad_length = window_size - n_timepoints
        data = np.pad(data, ((0, 0), (0, pad_length)), mode=pad_mode)
        n_windows = 1
    
    # Extract windows
    windows = []
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        
        if end <= data.shape[1]:
            windows.append(data[:, start:end])
        else:
            # Pad last window if needed
            window = data[:, start:]
            pad_length = window_size - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_length)), mode=pad_mode)
            windows.append(window)
    
    return np.array(windows)


def segment_batch(
    signals: np.ndarray,
    window_size: int,
    overlap: float = 0.5,
    labels: Optional[np.ndarray] = None,
    subject_ids: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment a batch of signals with labels.
    
    Args:
        signals: Batch of signals, shape (n_samples, n_channels, n_timepoints)
        window_size: Window size in timepoints
        overlap: Overlap fraction
        labels: Optional labels, shape (n_samples,)
        subject_ids: Optional subject IDs, shape (n_samples,)
        
    Returns:
        segmented_signals: Shape (n_total_windows, n_channels, window_size)
        segmented_labels: Shape (n_total_windows,) if labels provided
        segmented_subjects: Shape (n_total_windows,) if subject_ids provided
    """
    all_windows = []
    all_labels = []
    all_subjects = []
    
    for i in range(signals.shape[0]):
        windows = segment_signal(signals[i], window_size, overlap)
        all_windows.append(windows)
        
        if labels is not None:
            # Same label for all windows from this sample
            all_labels.extend([labels[i]] * len(windows))
            
        if subject_ids is not None:
            all_subjects.extend([subject_ids[i]] * len(windows))
    
    segmented = np.concatenate(all_windows, axis=0)
    
    out_labels = np.array(all_labels) if labels is not None else None
    out_subjects = np.array(all_subjects) if subject_ids is not None else None
    
    return segmented, out_labels, out_subjects


def downsample(
    data: np.ndarray,
    factor: int
) -> np.ndarray:
    """
    Downsample signal by integer factor.
    
    Uses anti-aliasing filter before decimation.
    
    Args:
        data: Input signal, shape (..., n_timepoints)
        factor: Downsampling factor
        
    Returns:
        Downsampled signal
    """
    from scipy.signal import decimate
    
    if data.ndim == 1:
        return decimate(data, factor, ftype='fir')
    
    # Handle multi-dimensional
    result = np.apply_along_axis(
        lambda x: decimate(x, factor, ftype='fir'),
        axis=-1,
        arr=data
    )
    
    return result


def upsample(
    data: np.ndarray,
    factor: int
) -> np.ndarray:
    """
    Upsample signal by integer factor using interpolation.
    
    Args:
        data: Input signal
        factor: Upsampling factor
        
    Returns:
        Upsampled signal
    """
    from scipy.signal import resample
    
    new_length = data.shape[-1] * factor
    
    if data.ndim == 1:
        return resample(data, new_length)
    
    return resample(data, new_length, axis=-1)


def remove_baseline(
    data: np.ndarray,
    baseline_window: Optional[Tuple[int, int]] = None,
    method: str = 'mean'
) -> np.ndarray:
    """
    Remove baseline from signal.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints)
        baseline_window: (start, end) indices for baseline period.
                        If None, uses entire signal.
        method: 'mean' or 'median'
        
    Returns:
        Baseline-corrected signal
    """
    if baseline_window is None:
        baseline = data
    else:
        start, end = baseline_window
        baseline = data[..., start:end]
    
    if method == 'mean':
        correction = np.mean(baseline, axis=-1, keepdims=True)
    else:
        correction = np.median(baseline, axis=-1, keepdims=True)
    
    return data - correction


def compute_statistics(
    data: np.ndarray,
    axis: int = -1
) -> dict:
    """
    Compute summary statistics of signal.
    
    Args:
        data: Input signal
        axis: Axis along which to compute statistics
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data, axis=axis),
        'std': np.std(data, axis=axis),
        'min': np.min(data, axis=axis),
        'max': np.max(data, axis=axis),
        'median': np.median(data, axis=axis),
        'skewness': stats.skew(data, axis=axis),
        'kurtosis': stats.kurtosis(data, axis=axis),
        'rms': np.sqrt(np.mean(data ** 2, axis=axis))
    }
