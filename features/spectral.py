"""
Spectral Feature Extraction

Implements STFT spectrograms, Power Spectral Density (PSD),
and frequency band power extraction for neural signals.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings


def get_frequency_bands() -> Dict[str, Tuple[float, float]]:
    """
    Get standard EEG frequency band definitions.
    
    Returns:
        Dictionary mapping band names to (low, high) frequency ranges in Hz
    """
    return {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 100.0)
    }


def compute_stft(
    data: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 64,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform to get spectrogram.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints) or (n_timepoints,)
        fs: Sampling frequency in Hz
        n_fft: FFT window size
        hop_length: Hop size between windows
        window: Window function ('hann', 'hamming', etc.)
        
    Returns:
        frequencies: Array of frequency bins
        times: Array of time points
        spectrogram: Complex STFT output, shape (n_channels, n_freq, n_time)
                    or (n_freq, n_time) for single channel
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    n_channels = data.shape[0]
    
    # Compute STFT for first channel to get output shape
    f, t, Zxx = signal.stft(
        data[0],
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=True
    )
    
    # Initialize output array
    spectrograms = np.zeros((n_channels, len(f), len(t)), dtype=np.complex128)
    spectrograms[0] = Zxx
    
    # Compute for remaining channels
    for ch in range(1, n_channels):
        _, _, Zxx = signal.stft(
            data[ch],
            fs=fs,
            window=window,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            return_onesided=True
        )
        spectrograms[ch] = Zxx
    
    if squeeze:
        spectrograms = spectrograms[0]
    
    return f, t, spectrograms


def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 64,
    window: str = 'hann',
    log_scale: bool = True,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnitude spectrogram (power spectrum over time).
    
    Args:
        data: Input signal
        fs: Sampling frequency
        n_fft: FFT size
        hop_length: Hop size
        window: Window function
        log_scale: If True, return log-power spectrogram
        eps: Small constant for log stability
        
    Returns:
        frequencies, times, spectrogram (magnitude)
    """
    f, t, Zxx = compute_stft(data, fs, n_fft, hop_length, window)
    
    # Compute magnitude
    mag = np.abs(Zxx)
    
    if log_scale:
        mag = np.log10(mag + eps)
    
    return f, t, mag


def compute_psd(
    data: np.ndarray,
    fs: float,
    method: str = 'welch',
    nperseg: int = 256,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints) or (n_timepoints,)
        fs: Sampling frequency
        method: PSD estimation method ('welch' or 'periodogram')
        nperseg: Segment length for Welch's method
        noverlap: Overlap between segments (default: nperseg // 2)
        
    Returns:
        frequencies: Frequency bins
        psd: Power spectral density, shape (n_channels, n_freq) or (n_freq,)
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    n_channels = data.shape[0]
    
    if method == 'welch':
        f, pxx = signal.welch(
            data[0], fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        
        psd = np.zeros((n_channels, len(f)))
        psd[0] = pxx
        
        for ch in range(1, n_channels):
            _, pxx = signal.welch(
                data[ch], fs=fs, nperseg=nperseg, noverlap=noverlap
            )
            psd[ch] = pxx
            
    else:  # periodogram
        f, pxx = signal.periodogram(data[0], fs=fs)
        
        psd = np.zeros((n_channels, len(f)))
        psd[0] = pxx
        
        for ch in range(1, n_channels):
            _, pxx = signal.periodogram(data[ch], fs=fs)
            psd[ch] = pxx
    
    if squeeze:
        psd = psd[0]
    
    return f, psd


def extract_band_power(
    data: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'welch',
    relative: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract power in specific frequency bands.
    
    Args:
        data: Input signal
        fs: Sampling frequency
        bands: Dictionary of band name -> (low, high) frequencies.
               Uses default EEG bands if None.
        method: PSD estimation method
        relative: If True, return relative power (percentage of total)
        
    Returns:
        Dictionary mapping band names to power values per channel
    """
    if bands is None:
        bands = get_frequency_bands()
    
    # Compute PSD
    f, psd = compute_psd(data, fs, method=method)
    
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
    
    n_channels = psd.shape[0]
    
    # Frequency resolution
    freq_res = f[1] - f[0] if len(f) > 1 else 1.0
    
    band_powers = {}
    
    # Total power for relative calculation
    total_power = np.sum(psd, axis=1) * freq_res if relative else None
    
    for band_name, (low, high) in bands.items():
        # Find frequency indices in band
        band_mask = (f >= low) & (f <= high)
        
        if not np.any(band_mask):
            warnings.warn(f"No frequencies found in band {band_name}: {low}-{high} Hz")
            band_powers[band_name] = np.zeros(n_channels)
            continue
        
        # Integrate PSD in band
        power = np.sum(psd[:, band_mask], axis=1) * freq_res
        
        if relative and total_power is not None:
            power = power / (total_power + 1e-10) * 100  # Percentage
        
        band_powers[band_name] = power
    
    return band_powers


def compute_spectrogram_features(
    data: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop_length: int = 64,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute spectrogram features suitable for CNN input.
    
    Returns averaged spectrogram across channels with normalization.
    
    Args:
        data: Input signal, shape (n_channels, n_timepoints)
        fs: Sampling frequency
        n_fft: FFT size
        hop_length: Hop size
        normalize: If True, apply z-score normalization
        
    Returns:
        spectrogram: Shape (1, n_freq, n_time) ready for CNN
    """
    f, t, spec = compute_spectrogram(
        data, fs, n_fft, hop_length, log_scale=True
    )
    
    # Average across channels
    if spec.ndim == 3:
        spec_mean = np.mean(spec, axis=0)
    else:
        spec_mean = spec
    
    if normalize:
        spec_mean = (spec_mean - np.mean(spec_mean)) / (np.std(spec_mean) + 1e-8)
    
    # Add channel dimension for CNN
    return spec_mean[np.newaxis, :, :]


def compute_temporal_features(
    spectrogram: np.ndarray,
    axis: int = -1
) -> Dict[str, np.ndarray]:
    """
    Compute temporal statistics from spectrogram.
    
    Args:
        spectrogram: Spectrogram array, shape (..., n_time)
        axis: Time axis
        
    Returns:
        Dictionary of temporal features
    """
    return {
        'mean': np.mean(spectrogram, axis=axis),
        'std': np.std(spectrogram, axis=axis),
        'max': np.max(spectrogram, axis=axis),
        'min': np.min(spectrogram, axis=axis),
        'skew': np.mean((spectrogram - np.mean(spectrogram, axis=axis, keepdims=True)) ** 3, axis=axis),
        'diff_mean': np.mean(np.abs(np.diff(spectrogram, axis=axis)), axis=axis)
    }


def compute_spectral_entropy(
    psd: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute spectral entropy from PSD.
    
    Higher entropy indicates more uniform power distribution.
    
    Args:
        psd: Power spectral density
        normalize: If True, normalize to [0, 1]
        
    Returns:
        Spectral entropy per channel
    """
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
    
    # Normalize PSD to probability distribution
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-10)
    
    # Compute entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
    
    if normalize:
        max_entropy = np.log2(psd.shape[1])
        entropy = entropy / max_entropy
    
    return entropy


def compute_spectral_edge_frequency(
    f: np.ndarray,
    psd: np.ndarray,
    percentage: float = 95.0
) -> np.ndarray:
    """
    Compute spectral edge frequency (SEF).
    
    The frequency below which a certain percentage of total power lies.
    
    Args:
        f: Frequency bins
        psd: Power spectral density
        percentage: Percentage of power (default 95%)
        
    Returns:
        SEF per channel
    """
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)
    
    n_channels = psd.shape[0]
    sef = np.zeros(n_channels)
    
    for ch in range(n_channels):
        cumsum = np.cumsum(psd[ch])
        total = cumsum[-1]
        threshold = total * (percentage / 100.0)
        
        idx = np.searchsorted(cumsum, threshold)
        idx = min(idx, len(f) - 1)
        sef[ch] = f[idx]
    
    return sef
