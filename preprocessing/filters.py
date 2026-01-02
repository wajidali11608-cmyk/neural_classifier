"""
Signal Filtering Utilities for Neural Activity Data

Implements bandpass filtering, notch filtering for powerline interference,
and artifact removal for EEG/neural signals.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional, Union
import warnings


def butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        b, a: Filter coefficients
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure frequencies are valid
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to neural signal data.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints) or (n_timepoints,)
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal with same shape as input
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    
    # Handle single channel
    if data.ndim == 1:
        return filtfilt(b, a, data)
    
    # Multi-channel: filter each channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    
    return filtered


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.
    
    Removes 50Hz or 60Hz powerline noise and harmonics.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints) or (n_timepoints,)
        notch_freq: Frequency to remove (typically 50 or 60 Hz)
        fs: Sampling frequency in Hz
        quality_factor: Quality factor (higher = narrower notch)
        
    Returns:
        Filtered signal
    """
    # Design notch filter
    w0 = notch_freq / (fs / 2)
    
    if w0 >= 1.0:
        warnings.warn(f"Notch frequency {notch_freq} Hz is at or above Nyquist. Skipping notch filter.")
        return data
    
    b, a = iirnotch(w0, quality_factor)
    
    # Handle single channel
    if data.ndim == 1:
        return filtfilt(b, a, data)
    
    # Multi-channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    
    return filtered


def apply_filters(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 100.0,
    notch_freq: Optional[float] = 50.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply complete filtering pipeline to neural signals.
    
    Applies bandpass filter followed by optional notch filter
    for powerline interference removal.
    
    Args:
        data: Input signal, shape (n_samples, n_channels, n_timepoints) or
              (n_channels, n_timepoints) or (n_timepoints,)
        fs: Sampling frequency in Hz
        lowcut: Low cutoff for bandpass
        highcut: High cutoff for bandpass
        notch_freq: Powerline frequency to notch out (None to skip)
        order: Butterworth filter order
        
    Returns:
        Filtered signal with same shape as input
    """
    # Handle batch dimension
    if data.ndim == 3:
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = apply_filters(
                data[i], fs, lowcut, highcut, notch_freq, order
            )
        return filtered
    
    # Apply bandpass
    filtered = bandpass_filter(data, lowcut, highcut, fs, order)
    
    # Apply notch filter if specified
    if notch_freq is not None:
        filtered = notch_filter(filtered, notch_freq, fs)
        
        # Also remove harmonics (2x, 3x)
        for harmonic in [2, 3]:
            harm_freq = notch_freq * harmonic
            if harm_freq < fs / 2:
                filtered = notch_filter(filtered, harm_freq, fs)
    
    return filtered


def artifact_removal(
    data: np.ndarray,
    threshold_std: float = 5.0,
    window_size: int = 256,
    method: str = 'interpolate'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove artifacts from neural signals based on amplitude thresholds.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints)
        threshold_std: Threshold in standard deviations
        window_size: Window size for local statistics
        method: 'interpolate' to replace artifacts, 'zero' to zero out,
                'nan' to mark as NaN
                
    Returns:
        cleaned_data: Artifact-corrected signal
        artifact_mask: Boolean mask indicating artifact locations
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_timepoints = data.shape
    cleaned = data.copy()
    artifact_mask = np.zeros_like(data, dtype=bool)
    
    for ch in range(n_channels):
        channel = data[ch]
        
        # Compute rolling statistics
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        
        # Identify artifacts
        threshold_high = mean_val + threshold_std * std_val
        threshold_low = mean_val - threshold_std * std_val
        
        artifacts = (channel > threshold_high) | (channel < threshold_low)
        artifact_mask[ch] = artifacts
        
        # Handle artifacts based on method
        if method == 'interpolate':
            # Linear interpolation over artifact regions
            artifact_indices = np.where(artifacts)[0]
            good_indices = np.where(~artifacts)[0]
            
            if len(good_indices) > 0 and len(artifact_indices) > 0:
                cleaned[ch, artifacts] = np.interp(
                    artifact_indices, 
                    good_indices, 
                    channel[good_indices]
                )
                
        elif method == 'zero':
            cleaned[ch, artifacts] = 0
            
        elif method == 'nan':
            cleaned[ch, artifacts] = np.nan
    
    return cleaned, artifact_mask


def compute_signal_quality(
    data: np.ndarray,
    fs: float
) -> dict:
    """
    Compute signal quality metrics.
    
    Args:
        data: Signal data, shape (n_channels, n_timepoints)
        fs: Sampling frequency
        
    Returns:
        Dictionary with quality metrics
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    metrics = {}
    
    # RMS per channel
    metrics['rms'] = np.sqrt(np.mean(data ** 2, axis=1))
    
    # Peak-to-peak amplitude
    metrics['peak_to_peak'] = np.ptp(data, axis=1)
    
    # Variance
    metrics['variance'] = np.var(data, axis=1)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(data), axis=1) != 0, axis=1)
    metrics['zero_crossing_rate'] = zero_crossings / data.shape[1]
    
    # Kurtosis (indicates artifact presence)
    from scipy.stats import kurtosis
    metrics['kurtosis'] = kurtosis(data, axis=1)
    
    # Mean quality score (lower kurtosis and reasonable variance is better)
    quality_scores = 1 / (1 + np.abs(metrics['kurtosis'] - 3))
    metrics['quality_score'] = np.mean(quality_scores)
    
    return metrics


def highpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply highpass filter to remove slow drifts.
    
    Args:
        data: Input signal
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    normalized_cutoff = max(normalized_cutoff, 0.001)
    
    b, a = butter(order, normalized_cutoff, btype='high')
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    
    return filtered


def lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply lowpass filter.
    
    Args:
        data: Input signal
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    normalized_cutoff = min(normalized_cutoff, 0.999)
    
    b, a = butter(order, normalized_cutoff, btype='low')
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    
    return filtered
