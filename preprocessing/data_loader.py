"""
Data Loading Utilities for Neural Activity Classification

Handles loading raw time-series data from CSV/other formats,
PyTorch Dataset creation, and subject-stratified data splitting.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings


class NeuralDataset(Dataset):
    """
    PyTorch Dataset for neural activity data.
    
    Handles multi-channel time-series with subject-level information
    to prevent data leakage during cross-validation.
    
    Args:
        signals: Neural signals of shape (n_samples, n_channels, n_timepoints)
        labels: Class labels (0=healthy, 1=schizophrenia, 2=bipolar)
        subject_ids: Subject IDs for stratification
        spectrograms: Optional precomputed spectrograms
        graphs: Optional precomputed graph data
        transform: Optional transform to apply
    """
    
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        spectrograms: Optional[np.ndarray] = None,
        graphs: Optional[List] = None,
        transform: Optional[callable] = None
    ):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids if subject_ids is not None else np.arange(len(signals))
        self.spectrograms = spectrograms
        self.graphs = graphs
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'signal': self.signals[idx],
            'label': self.labels[idx],
            'subject_id': self.subject_ids[idx]
        }
        
        if self.spectrograms is not None:
            sample['spectrogram'] = torch.FloatTensor(self.spectrograms[idx])
            
        if self.graphs is not None:
            sample['graph'] = self.graphs[idx]
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        class_counts = torch.bincount(self.labels)
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
        return weights
    
    def get_subject_indices(self) -> Dict[int, List[int]]:
        """Get sample indices grouped by subject."""
        subject_to_indices = {}
        for idx, subj_id in enumerate(self.subject_ids):
            if subj_id not in subject_to_indices:
                subject_to_indices[subj_id] = []
            subject_to_indices[subj_id].append(idx)
        return subject_to_indices


def load_neural_data(
    filepath: Union[str, Path],
    n_channels: int = 64,
    sample_rate: float = 256.0,
    label_column: str = 'label',
    subject_column: str = 'subject_id'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load neural activity data from CSV file.
    
    Expected CSV format:
    - Columns for each channel (ch_0, ch_1, ... or numeric indices)
    - 'label' column with class labels
    - 'subject_id' column with subject identifiers
    - Each row is a timepoint within a segment
    
    Args:
        filepath: Path to CSV file
        n_channels: Expected number of channels
        sample_rate: Sampling rate in Hz
        label_column: Name of label column
        subject_column: Name of subject ID column
        
    Returns:
        signals: Array of shape (n_samples, n_channels, n_timepoints)
        labels: Array of class labels
        subject_ids: Array of subject IDs
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Identify channel columns
    channel_cols = [c for c in df.columns if c.startswith('ch_') or c.isdigit()]
    if len(channel_cols) == 0:
        # Try to infer channel columns (exclude metadata columns)
        meta_cols = {label_column, subject_column, 'segment_id', 'timestamp', 'time'}
        channel_cols = [c for c in df.columns if c not in meta_cols]
    
    if len(channel_cols) < n_channels:
        warnings.warn(f"Found {len(channel_cols)} channels, expected {n_channels}")
    
    # Group by segment
    if 'segment_id' in df.columns:
        segments = df.groupby([subject_column, 'segment_id'])
    else:
        # Assume each unique (subject, label) combo is a segment
        df['segment_id'] = df.groupby([subject_column, label_column]).ngroup()
        segments = df.groupby('segment_id')
    
    signals_list = []
    labels_list = []
    subjects_list = []
    
    for seg_id, group in segments:
        # Extract channel data
        signal = group[channel_cols].values.T  # (n_channels, n_timepoints)
        signals_list.append(signal)
        
        # Get label and subject (should be same for all rows in segment)
        labels_list.append(group[label_column].iloc[0])
        subjects_list.append(group[subject_column].iloc[0])
    
    signals = np.array(signals_list)
    labels = np.array(labels_list)
    subject_ids = np.array(subjects_list)
    
    return signals, labels, subject_ids


def generate_synthetic_data(
    n_samples: int = 300,
    n_channels: int = 64,
    n_timepoints: int = 1024,
    sample_rate: float = 256.0,
    n_subjects: int = 30,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic neural activity data for testing.
    
    Creates data with class-specific frequency characteristics:
    - Healthy: Balanced alpha/beta activity
    - Schizophrenia: Reduced alpha, increased gamma, altered connectivity
    - Bipolar: Irregular theta/beta patterns
    
    Args:
        n_samples: Total number of samples
        n_channels: Number of channels
        n_timepoints: Timepoints per sample
        sample_rate: Sampling rate in Hz
        n_subjects: Number of unique subjects
        seed: Random seed
        
    Returns:
        signals: Shape (n_samples, n_channels, n_timepoints)
        labels: Shape (n_samples,)
        subject_ids: Shape (n_samples,)
    """
    np.random.seed(seed)
    
    n_per_class = n_samples // 3
    samples_per_subject = n_samples // n_subjects
    
    signals = []
    labels = []
    subject_ids = []
    
    t = np.linspace(0, n_timepoints / sample_rate, n_timepoints)
    
    for class_idx in range(3):
        for i in range(n_per_class):
            # Assign to a subject
            subject_id = (class_idx * n_per_class + i) % n_subjects
            
            # Generate base signal (sum of oscillations)
            signal = np.zeros((n_channels, n_timepoints))
            
            for ch in range(n_channels):
                # Common noise
                noise = np.random.randn(n_timepoints) * 0.5
                
                # Class-specific frequency patterns
                if class_idx == 0:  # Healthy
                    # Strong alpha (8-13 Hz), moderate beta
                    alpha = np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) * 2.0
                    beta = np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi) * 1.0
                    theta = np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi) * 0.8
                    gamma = np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi) * 0.3
                    
                elif class_idx == 1:  # Schizophrenia
                    # Reduced alpha, increased gamma, disrupted patterns
                    alpha = np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) * 0.8
                    beta = np.sin(2 * np.pi * 25 * t + np.random.rand() * 2 * np.pi) * 1.5
                    theta = np.sin(2 * np.pi * 5 * t + np.random.rand() * 2 * np.pi) * 1.2
                    gamma = np.sin(2 * np.pi * 45 * t + np.random.rand() * 2 * np.pi) * 1.5
                    # Add bursting pattern
                    burst = np.random.binomial(1, 0.1, n_timepoints) * np.random.randn(n_timepoints) * 2
                    noise += burst
                    
                else:  # Bipolar
                    # Irregular theta/beta, mood-dependent fluctuations
                    alpha = np.sin(2 * np.pi * 9 * t + np.random.rand() * 2 * np.pi) * 1.2
                    beta = np.sin(2 * np.pi * 22 * t + np.random.rand() * 2 * np.pi) * 1.8
                    theta = np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi) * 1.5
                    gamma = np.sin(2 * np.pi * 35 * t + np.random.rand() * 2 * np.pi) * 0.6
                    # Add slow modulation (mood cycles)
                    modulation = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
                    alpha *= modulation
                    beta *= modulation
                
                signal[ch] = alpha + beta + theta + gamma + noise
                
                # Add inter-channel correlation (spatial structure)
                if ch > 0:
                    correlation = 0.3 if class_idx == 0 else (0.1 if class_idx == 1 else 0.4)
                    signal[ch] += correlation * signal[ch - 1]
            
            signals.append(signal)
            labels.append(class_idx)
            subject_ids.append(subject_id)
    
    # Shuffle
    perm = np.random.permutation(len(signals))
    signals = np.array(signals)[perm]
    labels = np.array(labels)[perm]
    subject_ids = np.array(subject_ids)[perm]
    
    return signals, labels, subject_ids


def subject_stratified_split(
    signals: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data ensuring no subject appears in both train and test sets.
    
    This prevents data leakage by keeping all samples from the same
    subject in the same split.
    
    Args:
        signals: Neural signals
        labels: Class labels
        subject_ids: Subject identifiers
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    unique_subjects = np.unique(subject_ids)
    
    # Get majority label for each subject (for stratification)
    subject_labels = []
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_labels = labels[mask]
        majority_label = np.bincount(subj_labels).argmax()
        subject_labels.append(majority_label)
    subject_labels = np.array(subject_labels)
    
    # Split subjects
    train_val_subjs, test_subjs = train_test_split(
        unique_subjects, 
        test_size=test_size,
        stratify=subject_labels,
        random_state=seed
    )
    
    # Get labels for train_val subjects
    train_val_labels = subject_labels[np.isin(unique_subjects, train_val_subjs)]
    
    # Split train_val into train and val
    val_fraction = val_size / (1 - test_size)
    train_subjs, val_subjs = train_test_split(
        train_val_subjs,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed
    )
    
    # Create masks
    train_mask = np.isin(subject_ids, train_subjs)
    val_mask = np.isin(subject_ids, val_subjs)
    test_mask = np.isin(subject_ids, test_subjs)
    
    return {
        'train': (signals[train_mask], labels[train_mask], subject_ids[train_mask]),
        'val': (signals[val_mask], labels[val_mask], subject_ids[val_mask]),
        'test': (signals[test_mask], labels[test_mask], subject_ids[test_mask])
    }


def create_data_loaders(
    dataset: NeuralDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create PyTorch DataLoader from dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_kfold_splits(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get subject-stratified k-fold cross-validation splits.
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    unique_subjects = np.unique(subject_ids)
    
    # Get subject-level labels for stratification
    subject_labels = []
    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_label = np.bincount(labels[mask]).argmax()
        subject_labels.append(subj_label)
    subject_labels = np.array(subject_labels)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    for train_subj_idx, val_subj_idx in skf.split(unique_subjects, subject_labels):
        train_subjs = unique_subjects[train_subj_idx]
        val_subjs = unique_subjects[val_subj_idx]
        
        train_mask = np.isin(subject_ids, train_subjs)
        val_mask = np.isin(subject_ids, val_subjs)
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        splits.append((train_indices, val_indices))
    
    return splits
