"""
Uncertainty Estimation for Neural Classification

Implements Monte Carlo dropout, temperature scaling,
and calibration metrics for reliable predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def enable_dropout(model: nn.Module):
    """Enable dropout layers during inference for MC dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    data,
    n_samples: int = 30,
    device: str = 'cpu',
    is_hybrid: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo dropout prediction for uncertainty estimation.
    
    Performs multiple stochastic forward passes with dropout enabled
    to estimate prediction uncertainty.
    
    Args:
        model: Trained model with dropout layers
        data: Input data (batch dictionary or tensor)
        n_samples: Number of MC samples
        device: Device to use
        is_hybrid: If True, model is hybrid (needs spectrogram + graph)
        
    Returns:
        mean_probs: Mean predicted probabilities
        std_probs: Std of probabilities (uncertainty)
        predictions: Argmax predictions
    """
    model.eval()
    enable_dropout(model)  # Keep dropout active
    
    all_probs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            if is_hybrid:
                specs = data['spectrogram'].to(device)
                graphs = data['graph'].to(device)
                logits = model(specs, graphs)
            else:
                if isinstance(data, dict):
                    if 'spectrogram' in data:
                        inputs = data['spectrogram'].to(device)
                    else:
                        inputs = data['signal'].to(device)
                else:
                    inputs = data.to(device)
                
                logits = model(inputs)
            
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)  # (n_samples, batch, n_classes)
    
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)
    predictions = np.argmax(mean_probs, axis=-1)
    
    model.eval()  # Reset to eval mode
    
    return mean_probs, std_probs, predictions


def compute_predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute predictive entropy from probabilities.
    
    Higher entropy = more uncertainty.
    
    Args:
        probs: Probabilities, shape (batch, n_classes)
        
    Returns:
        Entropy per sample
    """
    return -np.sum(probs * np.log(probs + 1e-10), axis=-1)


def compute_mutual_information(all_probs: np.ndarray) -> np.ndarray:
    """
    Compute mutual information (epistemic uncertainty).
    
    Captures model uncertainty (vs. data uncertainty).
    
    Args:
        all_probs: MC samples, shape (n_samples, batch, n_classes)
        
    Returns:
        Mutual information per sample
    """
    # Mean of entropies
    mean_entropy = np.mean([
        compute_predictive_entropy(p) for p in all_probs
    ], axis=0)
    
    # Entropy of mean
    mean_probs = np.mean(all_probs, axis=0)
    entropy_of_mean = compute_predictive_entropy(mean_probs)
    
    # Mutual information = total uncertainty - aleatoric uncertainty
    return entropy_of_mean - mean_entropy


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    
    Learns a single temperature parameter to calibrate model outputs.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        device: str = 'cpu',
        max_iter: int = 50,
        lr: float = 0.01
    ):
        """
        Learn temperature parameter on validation set.
        
        Args:
            model: Trained model
            val_loader: Validation data loader
            device: Device
            max_iter: Optimization iterations
            lr: Learning rate
        """
        model.eval()
        self.to(device)
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)
                logits = model(inputs)
                
                all_logits.append(logits)
                all_labels.append(labels)
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_fn():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        
        print(f"Calibrated temperature: {self.temperature.item():.4f}")


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics: ECE and MCE.
    
    Expected Calibration Error (ECE): Average gap between confidence and accuracy
    Maximum Calibration Error (MCE): Maximum gap
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of confidence bins
        
    Returns:
        Dictionary with ECE, MCE, and per-bin data
    """
    confidences = np.max(y_prob, axis=-1)
    predictions = np.argmax(y_prob, axis=-1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    bin_data = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            gap = abs(avg_accuracy - avg_confidence)
            
            ece += prop_in_bin * gap
            mce = max(mce, gap)
            
            bin_data.append({
                'bin_lower': bin_boundaries[i],
                'bin_upper': bin_boundaries[i + 1],
                'confidence': avg_confidence,
                'accuracy': avg_accuracy,
                'count': in_bin.sum(),
                'proportion': prop_in_bin
            })
    
    return {
        'ece': ece,
        'mce': mce,
        'bin_data': bin_data
    }


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        bin_centers: Center of each bin
        bin_accuracies: Accuracy in each bin
        bin_counts: Number of samples in each bin
    """
    confidences = np.max(y_prob, axis=-1)
    predictions = np.argmax(y_prob, axis=-1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accuracies[i] = accuracies[in_bin].mean()
            bin_counts[i] = in_bin.sum()
    
    return bin_centers, bin_accuracies, bin_counts


class DeepEnsemble:
    """
    Deep ensemble for uncertainty estimation.
    
    Trains multiple models with different random seeds and
    aggregates their predictions.
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.n_models = len(models)
        
    def predict(
        self,
        data,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ensemble predictions with uncertainty.
        
        Returns:
            mean_probs: Mean predicted probabilities
            std_probs: Std of probabilities
            predictions: Argmax predictions
        """
        all_probs = []
        
        for model in self.models:
            model.eval()
            model.to(device)
            
            with torch.no_grad():
                if isinstance(data, dict):
                    inputs = data['spectrogram'].to(device)
                else:
                    inputs = data.to(device)
                
                logits = model(inputs)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.array(all_probs)
        
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        predictions = np.argmax(mean_probs, axis=-1)
        
        return mean_probs, std_probs, predictions


def confidence_calibration_rejection(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict[str, List]:
    """
    Analyze accuracy at different confidence rejection thresholds.
    
    Shows how accuracy improves when rejecting low-confidence predictions.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: Confidence thresholds for rejection
        
    Returns:
        Dictionary with accuracy and coverage at each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.95, 14).tolist()
    
    confidences = np.max(y_prob, axis=-1)
    predictions = np.argmax(y_prob, axis=-1)
    
    results = {'threshold': [], 'accuracy': [], 'coverage': []}
    
    for thresh in thresholds:
        mask = confidences >= thresh
        coverage = mask.mean()
        
        if coverage > 0:
            acc = (predictions[mask] == y_true[mask]).mean()
        else:
            acc = 0.0
        
        results['threshold'].append(thresh)
        results['accuracy'].append(acc)
        results['coverage'].append(coverage)
    
    return results
