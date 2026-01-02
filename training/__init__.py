"""Training module for neural classification."""
from .trainer import Trainer, train_epoch, validate_epoch
from .metrics import (
    compute_all_metrics, plot_confusion_matrix_data,
    compute_class_metrics, MetricTracker
)
from .uncertainty import (
    mc_dropout_predict, compute_calibration_metrics,
    TemperatureScaling
)

__all__ = [
    'Trainer', 'train_epoch', 'validate_epoch',
    'compute_all_metrics', 'plot_confusion_matrix_data',
    'compute_class_metrics', 'MetricTracker',
    'mc_dropout_predict', 'compute_calibration_metrics',
    'TemperatureScaling'
]
