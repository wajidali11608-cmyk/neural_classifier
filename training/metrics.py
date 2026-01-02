"""
Evaluation Metrics for Neural Classification

Comprehensive metrics including accuracy, precision, recall, F1,
AUROC, and confusion matrix computation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUROC)
        class_names: Names of classes
        average: Averaging method ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUROC (requires probabilities)
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:
                # Multi-class AUROC
                metrics['auroc'] = roc_auc_score(
                    y_true, y_prob, 
                    multi_class='ovr', 
                    average=average
                )
            else:
                # Binary AUROC
                metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics['auroc'] = 0.0
    
    # Balanced accuracy
    unique_classes = np.unique(y_true)
    per_class_acc = []
    for c in unique_classes:
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append((y_pred[mask] == c).mean())
    metrics['balanced_accuracy'] = np.mean(per_class_acc)
    
    return metrics


def compute_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        Dictionary mapping class names to their metrics
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if class_names is None:
        class_names = [f'class_{i}' for i in unique_classes]
    
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        if i >= len(unique_classes):
            continue
            
        mask_true = y_true == unique_classes[i]
        mask_pred = y_pred == unique_classes[i]
        
        tp = ((mask_true) & (mask_pred)).sum()
        fp = ((~mask_true) & (mask_pred)).sum()
        fn = ((mask_true) & (~mask_pred)).sum()
        tn = ((~mask_true) & (~mask_pred)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(mask_true.sum()),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    return class_metrics


def plot_confusion_matrix_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix data for plotting.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        class_names: Names of classes
        normalize: If True, return normalized matrix
        
    Returns:
        cm: Confusion matrix
        class_names: Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    return cm, class_names


def compute_roc_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute ROC curve data for each class.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities, shape (n_samples, n_classes)
        n_classes: Number of classes
        
    Returns:
        Dictionary mapping class index to fpr, tpr, thresholds
    """
    from sklearn.metrics import roc_curve
    
    roc_data = {}
    
    # One-hot encode true labels
    y_true_onehot = np.eye(n_classes)[y_true]
    
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_data[i] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc_score(y_true_onehot[:, i], y_prob[:, i])
        }
    
    return roc_data


def compute_pr_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute Precision-Recall curve data for each class.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        n_classes: Number of classes
        
    Returns:
        Dictionary with precision, recall, thresholds per class
    """
    pr_data = {}
    
    # One-hot encode
    y_true_onehot = np.eye(n_classes)[y_true]
    
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(
            y_true_onehot[:, i], y_prob[:, i]
        )
        pr_data[i] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'ap': average_precision_score(y_true_onehot[:, i], y_prob[:, i])
        }
    
    return pr_data


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate formatted classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        Formatted report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names
        self.reset()
        
    def reset(self):
        """Reset all tracked values."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
        
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """Update with batch predictions."""
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy().tolist())
            
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated predictions."""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities) if self.probabilities else None
        
        metrics = compute_all_metrics(y_true, y_pred, y_prob, self.class_names)
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        return confusion_matrix(y_true, y_pred)
