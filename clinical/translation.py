"""
Dataset Adaptation for Different Neural Modalities

Provides adapters to configure the pipeline for organoid, EEG,
or fMRI data with modality-specific preprocessing and parameters.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModalityConfig:
    """Configuration for a specific neural data modality."""
    name: str
    sample_rate: float
    n_channels: int
    frequency_range: Tuple[float, float]
    default_window_size: float  # seconds
    typical_artifacts: list
    preprocessing_notes: str


def get_modality_config(modality: str) -> ModalityConfig:
    """
    Get default configuration for a data modality.
    
    Args:
        modality: 'organoid', 'eeg', or 'fmri'
        
    Returns:
        ModalityConfig with appropriate settings
    """
    configs = {
        'organoid': ModalityConfig(
            name='Brain Organoid',
            sample_rate=20000.0,  # Typical MEA recording
            n_channels=60,  # 60-electrode MEA
            frequency_range=(0.1, 5000.0),  # Wide range for spikes
            default_window_size=2.0,
            typical_artifacts=['electrode_drift', 'electrical_noise'],
            preprocessing_notes='High-pass filter at 300Hz for spike detection'
        ),
        'eeg': ModalityConfig(
            name='EEG',
            sample_rate=256.0,  # Standard EEG
            n_channels=64,  # 64-channel EEG cap
            frequency_range=(0.5, 100.0),
            default_window_size=4.0,
            typical_artifacts=['eye_blinks', 'muscle', 'powerline'],
            preprocessing_notes='Common average reference, 50/60Hz notch filter'
        ),
        'fmri': ModalityConfig(
            name='fMRI',
            sample_rate=0.5,  # TR = 2s
            n_channels=100,  # ROIs from parcellation
            frequency_range=(0.01, 0.1),  # Hemodynamic frequencies
            default_window_size=60.0,  # 1 minute windows
            typical_artifacts=['motion', 'physiological_noise'],
            preprocessing_notes='Bandpass 0.01-0.1Hz, motion regression'
        )
    }
    
    if modality not in configs:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(configs.keys())}")
    
    return configs[modality]


class DatasetAdapter(ABC):
    """
    Abstract base class for modality-specific data adaptation.
    
    Subclass this to handle different data types (organoid, EEG, fMRI).
    """
    
    @abstractmethod
    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from file."""
        pass
    
    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply modality-specific preprocessing."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for models and training."""
        pass


class OrganoidAdapter(DatasetAdapter):
    """
    Adapter for brain organoid MEA recordings.
    
    Brain organoids produce high-frequency spike activity
    recorded on multi-electrode arrays (MEAs).
    """
    
    def __init__(
        self,
        sample_rate: float = 20000.0,
        n_channels: int = 60,
        spike_threshold: float = -4.0  # Standard deviations
    ):
        self.config = get_modality_config('organoid')
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.spike_threshold = spike_threshold
        
    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load organoid MEA data.
        
        Expects CSV or HDF5 format with voltage traces.
        """
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        # Extract channels
        channel_cols = [c for c in df.columns if c.startswith('ch_') or c.startswith('electrode_')]
        
        # Group by sample/recording
        if 'sample_id' in df.columns:
            groups = df.groupby('sample_id')
        else:
            # Single recording
            signals = df[channel_cols].values.T[np.newaxis, :, :]
            labels = np.array([df['label'].iloc[0]])
            subject_ids = np.array([0])
            return signals, labels, subject_ids
        
        signals_list = []
        labels_list = []
        subject_ids_list = []
        
        for sample_id, group in groups:
            signal = group[channel_cols].values.T
            signals_list.append(signal)
            labels_list.append(group['label'].iloc[0])
            subject_ids_list.append(group['subject_id'].iloc[0] if 'subject_id' in group else sample_id)
        
        return np.array(signals_list), np.array(labels_list), np.array(subject_ids_list)
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess organoid data.
        
        - High-pass filter at 300Hz (for spikes)
        - Artifact rejection
        - Downsampling for efficiency
        """
        from ..preprocessing.filters import bandpass_filter, artifact_removal
        
        # High-pass filter to isolate spikes
        filtered = bandpass_filter(data, 300.0, 5000.0, self.sample_rate, order=4)
        
        # Artifact rejection
        cleaned, _ = artifact_removal(filtered, threshold_std=5.0)
        
        # Downsample for analysis (optional)
        # Could use scipy.signal.decimate
        
        return cleaned
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration optimized for organoid data."""
        return {
            'data': {
                'sample_rate': self.sample_rate,
                'n_channels': self.n_channels,
                'window_size': 1.0,  # 1 second windows
                'lowcut': 300.0,
                'highcut': 5000.0,
            },
            'spectral': {
                'n_fft': 512,
                'hop_length': 128,
                'fmax': 5000.0
            },
            'model': {
                'use_spike_features': True,
                'use_burst_detection': True
            }
        }


class EEGAdapter(DatasetAdapter):
    """
    Adapter for EEG recordings.
    
    Handles standard clinical EEG with focus on oscillatory patterns.
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        n_channels: int = 64,
        reference: str = 'average'  # 'average', 'linked', 'laplacian'
    ):
        self.config = get_modality_config('eeg')
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.reference = reference
        
    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load EEG data.
        
        Supports common EEG formats (EDF, BDF, CSV).
        """
        import pandas as pd
        from pathlib import Path
        
        ext = Path(filepath).suffix.lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath)
            channel_cols = [c for c in df.columns if c.startswith('ch_') or 
                           c in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']]
            
            # Simplified loading
            groups = df.groupby(['subject_id', 'segment_id']) if 'segment_id' in df.columns else [(0, df)]
            
            signals_list = []
            labels_list = []
            subject_ids_list = []
            
            for (subj, seg), group in groups:
                signal = group[channel_cols].values.T
                signals_list.append(signal)
                labels_list.append(group['label'].iloc[0])
                subject_ids_list.append(subj)
            
            return np.array(signals_list), np.array(labels_list), np.array(subject_ids_list)
        
        else:
            raise NotImplementedError(f"Format {ext} not yet supported. Use CSV.")
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data.
        
        - Reference to common average
        - Bandpass 0.5-100 Hz
        - Notch filter at 50/60 Hz
        - Artifact rejection
        """
        from ..preprocessing.filters import apply_filters
        
        # Apply common average reference
        if self.reference == 'average':
            avg = np.mean(data, axis=-2, keepdims=True)
            data = data - avg
        
        # Filtering
        filtered = apply_filters(
            data, self.sample_rate,
            lowcut=0.5, highcut=100.0, notch_freq=50.0
        )
        
        return filtered
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration optimized for EEG."""
        return {
            'data': {
                'sample_rate': self.sample_rate,
                'n_channels': self.n_channels,
                'window_size': 4.0,
                'lowcut': 0.5,
                'highcut': 100.0,
            },
            'spectral': {
                'n_fft': 256,
                'hop_length': 64,
                'fmax': 100.0
            },
            'model': {
                'use_band_features': True,
                'focus_bands': ['alpha', 'beta', 'gamma']
            }
        }


class fMRIAdapter(DatasetAdapter):
    """
    Adapter for fMRI BOLD signal analysis.
    
    Handles parcellated ROI time series with slow hemodynamic dynamics.
    """
    
    def __init__(
        self,
        tr: float = 2.0,  # Repetition time in seconds
        n_rois: int = 100  # Number of brain regions
    ):
        self.config = get_modality_config('fmri')
        self.tr = tr
        self.sample_rate = 1.0 / tr
        self.n_rois = n_rois
        
    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load fMRI ROI time series.
        
        Expects CSV with ROI_0, ROI_1, ... columns.
        """
        import pandas as pd
        
        df = pd.read_csv(filepath)
        
        roi_cols = [c for c in df.columns if c.startswith('ROI_') or c.startswith('region_')]
        
        groups = df.groupby('subject_id')
        
        signals_list = []
        labels_list = []
        subject_ids_list = []
        
        for subj_id, group in groups:
            signal = group[roi_cols].values.T  # (n_rois, n_timepoints)
            signals_list.append(signal)
            labels_list.append(group['label'].iloc[0])
            subject_ids_list.append(subj_id)
        
        return np.array(signals_list), np.array(labels_list), np.array(subject_ids_list)
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess fMRI data.
        
        - Detrend
        - Bandpass 0.01-0.1 Hz
        - Z-score normalization per ROI
        """
        from scipy.signal import detrend
        from ..preprocessing.normalization import z_score_normalize
        from ..preprocessing.filters import bandpass_filter
        
        # Detrend
        detrended = detrend(data, axis=-1)
        
        # Bandpass for hemodynamic frequencies
        filtered = bandpass_filter(
            detrended, 0.01, 0.1, self.sample_rate, order=2
        )
        
        # Z-score
        normalized = z_score_normalize(filtered)
        
        return normalized
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration optimized for fMRI."""
        return {
            'data': {
                'sample_rate': self.sample_rate,
                'n_channels': self.n_rois,
                'window_size': 60.0,  # 1 minute windows
                'lowcut': 0.01,
                'highcut': 0.1,
            },
            'spectral': {
                'n_fft': 64,
                'hop_length': 16,
                'fmax': 0.1
            },
            'model': {
                'use_connectivity_only': True,  # Focus on connectivity for fMRI
                'gnn_emphasized': True
            }
        }


def adapt_pipeline(
    modality: str,
    base_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Adapt pipeline configuration for a specific modality.
    
    Args:
        modality: 'organoid', 'eeg', or 'fmri'
        base_config: Base configuration to modify
        
    Returns:
        Modified configuration dictionary
    """
    modality_config = get_modality_config(modality)
    
    if base_config is None:
        base_config = {}
    
    # Create adapter
    adapters = {
        'organoid': OrganoidAdapter,
        'eeg': EEGAdapter,
        'fmri': fMRIAdapter
    }
    
    adapter = adapters[modality]()
    adapter_config = adapter.get_config()
    
    # Merge configurations
    merged = {**base_config}
    for key, value in adapter_config.items():
        if key in merged and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    
    merged['modality'] = modality
    merged['modality_info'] = {
        'name': modality_config.name,
        'sample_rate': modality_config.sample_rate,
        'frequency_range': modality_config.frequency_range,
        'notes': modality_config.preprocessing_notes
    }
    
    return merged
