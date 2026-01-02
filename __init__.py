"""
Neural Activity Classification Pipeline

A comprehensive machine learning pipeline for classifying neural signals
(EEG, organoid, fMRI) into healthy, schizophrenia, and bipolar disorder.

Modules:
    config: Configuration dataclasses
    preprocessing: Data loading, filtering, normalization
    features: Spectral, coherence, and graph features
    models: CNN, GNN, and hybrid architectures
    training: Training loops, metrics, uncertainty estimation
    explainability: Grad-CAM, attention visualization
    visualization: Plotting utilities
    clinical: Dataset adapters, drug simulation
"""

from .config import Config, get_config

__version__ = "1.0.0"
__author__ = "Neural Classifier Team"

__all__ = ['Config', 'get_config']
