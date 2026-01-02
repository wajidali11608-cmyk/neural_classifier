"""Visualization module for neural classification."""
from .plots import (
    plot_spectrogram, plot_confusion_matrix, plot_saliency_map,
    plot_connectivity_graph, plot_calibration, plot_roc_curves,
    plot_training_history, plot_band_importance,
    plot_reliability_diagram, plot_feature_importance
)

__all__ = [
    'plot_spectrogram', 'plot_confusion_matrix', 'plot_saliency_map',
    'plot_connectivity_graph', 'plot_calibration', 'plot_roc_curves',
    'plot_training_history', 'plot_band_importance',
    'plot_reliability_diagram', 'plot_feature_importance'
]
