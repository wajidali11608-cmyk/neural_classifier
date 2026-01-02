"""
Training Loop and Cross-Validation

Implements training loop with early stopping, learning rate scheduling,
and stratified k-fold cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from pathlib import Path
import logging
from tqdm import tqdm
import json

from .metrics import MetricTracker, compute_all_metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cpu',
    is_hybrid: bool = False,
    grad_clip: Optional[float] = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        is_hybrid: If True, expect both spectrogram and graph inputs
        grad_clip: Gradient clipping norm (None to disable)
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    tracker = MetricTracker()
    
    for batch in tqdm(train_loader, desc='Training', leave=False):
        optimizer.zero_grad()
        
        if is_hybrid:
            # Hybrid model expects spectrogram and graph
            specs = batch['spectrogram'].to(device)
            graphs = batch['graph'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(specs, graphs)
        else:
            # Single modality
            if 'spectrogram' in batch:
                inputs = batch['spectrogram'].to(device)
            elif 'graph' in batch:
                inputs = batch['graph'].to(device)
            else:
                inputs = batch['signal'].to(device)
            
            labels = batch['label'].to(device)
            
            if hasattr(inputs, 'x'):  # PyG graph
                logits = model(inputs)
            else:
                logits = model(inputs)
        
        loss = criterion(logits, labels)
        loss.backward()
        
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        tracker.update(preds, labels, probs, loss.item())
    
    return tracker.compute()


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = 'cpu',
    is_hybrid: bool = False
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        is_hybrid: If True, expect hybrid inputs
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    tracker = MetricTracker()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            if is_hybrid:
                specs = batch['spectrogram'].to(device)
                graphs = batch['graph'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(specs, graphs)
            else:
                if 'spectrogram' in batch:
                    inputs = batch['spectrogram'].to(device)
                elif 'graph' in batch:
                    inputs = batch['graph'].to(device)
                else:
                    inputs = batch['signal'].to(device)
                
                labels = batch['label'].to(device)
                
                if hasattr(inputs, 'x'):
                    logits = model(inputs)
                else:
                    logits = model(inputs)
            
            loss = criterion(logits, labels)
            
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            tracker.update(preds, labels, probs, loss.item())
    
    return tracker.compute()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Complete training pipeline with validation and model saving.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        is_hybrid: If True, model expects hybrid inputs
        save_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        is_hybrid: bool = False,
        save_dir: Optional[str] = None,
        scheduler: Optional[object] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_hybrid = is_hybrid
        self.save_dir = Path(save_dir) if save_dir else None
        self.scheduler = scheduler
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        self.best_val_score = 0.0
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        patience: int = 15,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            n_epochs: Maximum epochs
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history
        """
        early_stop = EarlyStopping(patience=patience, mode='max')
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = train_epoch(
                self.model, train_loader, self.optimizer,
                self.criterion, self.device, self.is_hybrid
            )
            
            # Validate
            val_metrics = validate_epoch(
                self.model, val_loader,
                self.criterion, self.device, self.is_hybrid
            )
            
            # Update scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # Save best model
            if val_metrics['f1_score'] > self.best_val_score:
                self.best_val_score = val_metrics['f1_score']
                if self.save_dir:
                    self.save_checkpoint('best_model.pt')
            
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            
            # Early stopping
            if early_stop(val_metrics['f1_score']):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_score = checkpoint['best_val_score']
        self.history = checkpoint['history']
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on test set.
        
        Returns:
            metrics: Dictionary of metrics
            y_true: True labels
            y_pred: Predicted labels
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                if self.is_hybrid:
                    specs = batch['spectrogram'].to(self.device)
                    graphs = batch['graph'].to(self.device)
                    labels = batch['label'].to(self.device)
                    logits = self.model(specs, graphs)
                else:
                    if 'spectrogram' in batch:
                        inputs = batch['spectrogram'].to(self.device)
                    else:
                        inputs = batch['signal'].to(self.device)
                    labels = batch['label'].to(self.device)
                    logits = self.model(inputs)
                
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        
        return metrics, y_true, y_pred


def run_kfold_cv(
    model_fn: Callable,
    dataset,
    n_folds: int = 5,
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cpu',
    **trainer_kwargs
) -> Dict:
    """
    Run stratified k-fold cross-validation.
    
    Args:
        model_fn: Function that returns a fresh model instance
        dataset: Full dataset
        n_folds: Number of folds
        n_epochs: Epochs per fold
        batch_size: Batch size
        device: Device
        **trainer_kwargs: Additional Trainer arguments
        
    Returns:
        Dictionary with per-fold and mean metrics
    """
    from sklearn.model_selection import StratifiedKFold
    
    # Get labels for stratification
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        
        # Create fresh model
        model = model_fn()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        trainer = Trainer(
            model, optimizer, criterion,
            device=device,
            **trainer_kwargs
        )
        
        trainer.fit(train_loader, val_loader, n_epochs=n_epochs)
        
        # Evaluate
        metrics, _, _ = trainer.evaluate(val_loader)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold+1} Results: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    # Compute mean and std
    mean_metrics = {}
    std_metrics = {}
    
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        mean_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    print(f"\n{'='*50}")
    print("Cross-Validation Results")
    print(f"{'='*50}")
    for key in mean_metrics:
        print(f"{key}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    return {
        'fold_metrics': fold_metrics,
        'mean': mean_metrics,
        'std': std_metrics
    }
