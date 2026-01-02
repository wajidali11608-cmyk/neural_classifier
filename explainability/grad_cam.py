"""
Grad-CAM and Saliency Maps for CNN Explainability

Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
and input gradient saliency for interpreting CNN predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from scipy.ndimage import gaussian_filter


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Highlights important regions in spectrograms that contribute
    to classification decisions.
    
    Args:
        model: CNN model
        target_layer: Layer to extract activations from
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None
    ):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        if self.target_layer is None:
            # Find last conv layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.target_layer = module
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input spectrogram, shape (1, channels, H, W)
            target_class: Class to explain (None = predicted class)
            normalize: Normalize heatmap to [0, 1]
            
        Returns:
            cam: Grad-CAM heatmap, shape (H, W)
            predicted_class: Model's prediction
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Enable gradients
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=-1)
        
        predicted_class = output.argmax(dim=-1).item()
        confidence = probs[0, predicted_class].item()
        
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep positive contributions
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        cam = cam.squeeze().cpu().numpy()
        
        if normalize:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, predicted_class, confidence
    
    def explain_batch(
        self,
        batch_tensor: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """Generate Grad-CAM for a batch of inputs."""
        cams = []
        
        for i in range(batch_tensor.shape[0]):
            input_single = batch_tensor[i:i+1]
            target = target_classes[i] if target_classes else None
            cam, _, _ = self(input_single, target)
            cams.append(cam)
        
        return cams


def compute_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    smooth: bool = True,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Compute input gradient saliency map.
    
    Shows which input features most affect the prediction.
    
    Args:
        model: CNN model
        input_tensor: Input, shape (1, C, H, W)
        target_class: Class to explain (None = predicted)
        smooth: Apply Gaussian smoothing
        sigma: Smoothing parameter
        
    Returns:
        saliency: Saliency map, shape (H, W)
    """
    model.eval()
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=-1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients
    saliency = input_tensor.grad.abs()
    
    # Take max across channels
    saliency = saliency.max(dim=1)[0]
    
    saliency = saliency.squeeze().cpu().numpy()
    
    if smooth:
        saliency = gaussian_filter(saliency, sigma=sigma)
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


def get_class_activation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str
) -> np.ndarray:
    """
    Get activation maps from a specific layer.
    
    Args:
        model: CNN model
        input_tensor: Input tensor
        layer_name: Name of layer to extract
        
    Returns:
        Activation maps
    """
    activations = {}
    
    def hook(module, input, output):
        activations['target'] = output.detach()
    
    # Find and hook target layer
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    handle.remove()
    
    return activations.get('target', None).cpu().numpy()


class IntegratedGradients:
    """
    Integrated Gradients for attribution.
    
    More accurate than simple gradients by integrating along
    the path from baseline to input.
    """
    
    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps
        
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute integrated gradients.
        
        Args:
            input_tensor: Input
            target_class: Target class
            baseline: Baseline input (default: zeros)
            
        Returns:
            Attribution map
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        self.model.eval()
        
        # Compute path
        scaled_inputs = [
            baseline + (float(i) / self.n_steps) * (input_tensor - baseline)
            for i in range(self.n_steps + 1)
        ]
        
        # Compute gradients for each point
        grads = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad = True
            
            output = self.model(scaled_input)
            if target_class is None:
                target_class = output.argmax(dim=-1).item()
            
            self.model.zero_grad()
            output[0, target_class].backward()
            
            grads.append(scaled_input.grad.clone())
        
        # Average gradients
        avg_grads = torch.stack(grads, dim=0).mean(dim=0)
        
        # Compute integrated gradients
        ig = (input_tensor - baseline) * avg_grads
        
        # Sum over channels
        ig = ig.abs().sum(dim=1).squeeze().cpu().numpy()
        
        # Normalize
        ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
        
        return ig


def extract_spectral_biomarkers(
    saliency: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Extract interpretable biomarkers from saliency map.
    
    Identifies which frequency bands and time windows are important.
    
    Args:
        saliency: Saliency map, shape (n_freq, n_time)
        frequencies: Frequency axis values
        times: Time axis values
        threshold: Importance threshold
        
    Returns:
        Dictionary of biomarkers
    """
    # Find important regions
    important_mask = saliency > threshold
    
    biomarkers = {}
    
    # Frequency importance
    freq_importance = saliency.mean(axis=1)
    biomarkers['peak_frequency'] = frequencies[freq_importance.argmax()]
    biomarkers['frequency_importance'] = dict(zip(frequencies.tolist(), freq_importance.tolist()))
    
    # Band importance
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    band_importance = {}
    for band_name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies <= high)
        if mask.sum() > 0:
            band_importance[band_name] = float(freq_importance[mask].mean())
    
    biomarkers['band_importance'] = band_importance
    
    # Temporal dynamics
    time_importance = saliency.mean(axis=0)
    biomarkers['peak_time'] = times[time_importance.argmax()]
    biomarkers['temporal_variance'] = float(time_importance.var())
    
    return biomarkers
