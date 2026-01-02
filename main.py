"""
Neural Activity Classification Pipeline
========================================

Main orchestration script for classifying neural signals into:
- Healthy
- Schizophrenia
- Bipolar Disorder

Supports organoid, EEG, and fMRI data modalities.

Usage:
    python main.py --demo                    # Run demo with synthetic data
    python main.py --data path/to/data.csv   # Train on real data
    python main.py --help                    # Show all options
"""

import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline components
from config import Config, get_config
from preprocessing import (
    generate_synthetic_data, NeuralDataset, 
    subject_stratified_split, apply_filters, z_score_normalize,
    segment_signal
)
from features import (
    compute_stft, compute_spectrogram_features, 
    compute_coherence_matrix, build_connectivity_graph
)
from models import SpectrogramCNN, ConnectivityGNN, HybridClassifier
from training import Trainer, compute_all_metrics, MetricTracker
from training.metrics import plot_confusion_matrix_data, compute_roc_data
from training.uncertainty import mc_dropout_predict, compute_calibration_metrics
from explainability import GradCAM, extract_biomarkers
from visualization import (
    plot_spectrogram, plot_confusion_matrix, plot_saliency_map,
    plot_connectivity_graph, plot_calibration, plot_roc_curves,
    plot_training_history, plot_band_importance
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural Activity Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --data data/signals.csv --epochs 100
  python main.py --data data/eeg.csv --modality eeg --model hybrid
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data')
    parser.add_argument('--data', type=str, default='data/sample_data.csv',
                        help='Path to data file (CSV format). Defaults to data/sample_data.csv')
    parser.add_argument('--modality', type=str, default='eeg',
                        choices=['organoid', 'eeg', 'fmri'],
                        help='Data modality (default: eeg)')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'gnn', 'hybrid'],
                        help='Model architecture (default: cnn)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory (default: outputs)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def prepare_demo_data(config: Config, output_dir: Path):
    """Generate and prepare synthetic demo data."""
    print("\n" + "="*60)
    print("DEMO MODE: Generating Synthetic Neural Data")
    print("="*60)
    
    # Generate synthetic data
    signals, labels, subject_ids = generate_synthetic_data(
        n_samples=300,
        n_channels=config.data.n_channels,
        n_timepoints=1024,
        sample_rate=config.data.sample_rate,
        n_subjects=30,
        seed=config.seed
    )
    
    print(f"Generated {len(signals)} samples")
    print(f"  Channels: {signals.shape[1]}")
    print(f"  Timepoints: {signals.shape[2]}")
    print(f"  Classes: {np.bincount(labels)}")
    
    # Preprocess
    print("\nPreprocessing signals...")
    for i in range(len(signals)):
        signals[i] = apply_filters(
            signals[i], 
            config.data.sample_rate,
            lowcut=config.data.lowcut,
            highcut=config.data.highcut,
            notch_freq=config.data.notch_freq
        )
        signals[i] = z_score_normalize(signals[i])
    
    return signals, labels, subject_ids


def compute_features(
    signals: np.ndarray,
    config: Config,
    device: str
):
    """Compute spectrograms and graphs from signals."""
    print("\nExtracting features...")
    
    spectrograms = []
    graphs = []
    
    for i in range(len(signals)):
        # Spectrogram
        spec = compute_spectrogram_features(
            signals[i],
            config.data.sample_rate,
            n_fft=config.spectral.n_fft,
            hop_length=config.spectral.hop_length
        )
        spectrograms.append(spec)
        
        # Graph
        graph = build_connectivity_graph(
            signals[i],
            config.data.sample_rate,
            coherence_threshold=config.graph.coherence_threshold,
            device=device
        )
        graphs.append(graph)
    
    spectrograms = np.array(spectrograms)
    
    print(f"  Spectrogram shape: {spectrograms.shape}")
    print(f"  Graphs: {len(graphs)} samples")
    
    return spectrograms, graphs


class HybridDataset(torch.utils.data.Dataset):
    """Dataset for hybrid model with both spectrograms and graphs."""
    
    def __init__(self, spectrograms, graphs, labels):
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.graphs = graphs
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'spectrogram': self.spectrograms[idx],
            'graph': self.graphs[idx],
            'label': self.labels[idx]
        }


def train_model(
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    config: Config,
    args,
    output_dir: Path
):
    """Train the model."""
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    device = config.device
    model = model.to(device)
    
    # Create data loaders
    train_dataset = HybridDataset(
        train_data['spectrograms'],
        train_data['graphs'],
        train_data['labels']
    )
    val_dataset = HybridDataset(
        val_data['spectrograms'],
        val_data['graphs'],
        val_data['labels']
    )
    
    # Custom collate for graphs
    from torch_geometric.data import Batch
    
    def collate_fn(batch):
        specs = torch.stack([b['spectrogram'] for b in batch])
        graphs = Batch.from_data_list([b['graph'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        return {'spectrogram': specs, 'graph': graphs, 'label': labels}
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    # Optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            specs = batch['spectrogram'].to(device)
            graphs = batch['graph'].to(device)
            labels = batch['label'].to(device)
            
            if args.model == 'cnn':
                logits = model(specs)
            elif args.model == 'gnn':
                logits = model(graphs)
            else:
                logits = model(specs, graphs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                specs = batch['spectrogram'].to(device)
                graphs = batch['graph'].to(device)
                labels = batch['label'].to(device)
                
                if args.model == 'cnn':
                    logits = model(specs)
                elif args.model == 'gnn':
                    logits = model(graphs)
                else:
                    logits = model(specs, graphs)
                
                loss = criterion(logits, labels)
                
                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
        
        # Compute metrics
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        val_metrics = compute_all_metrics(
            np.array(val_labels), 
            np.array(val_preds),
            np.array(val_probs)
        )
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(np.mean(val_losses))
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        
        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train - Loss: {np.mean(train_losses):.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {np.mean(val_losses):.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    
    return history, val_labels, val_preds, val_probs


def evaluate_and_visualize(
    model: nn.Module,
    test_data: dict,
    config: Config,
    args,
    output_dir: Path
):
    """Evaluate model and generate visualizations."""
    print("\n" + "="*60)
    print("EVALUATION & VISUALIZATION")
    print("="*60)
    
    device = config.device
    class_names = config.data.class_names
    
    # Create test loader
    test_dataset = HybridDataset(
        test_data['spectrograms'],
        test_data['graphs'],
        test_data['labels']
    )
    
    from torch_geometric.data import Batch
    
    def collate_fn(batch):
        specs = torch.stack([b['spectrogram'] for b in batch])
        graphs = Batch.from_data_list([b['graph'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        return {'spectrogram': specs, 'graph': graphs, 'label': labels}
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Get predictions
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            specs = batch['spectrogram'].to(device)
            graphs = batch['graph'].to(device)
            labels = batch['label'].to(device)
            
            if args.model == 'cnn':
                logits = model(specs)
            elif args.model == 'gnn':
                logits = model(graphs)
            else:
                logits = model(specs, graphs)
            
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
    
    print("\nTest Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if 'auroc' in metrics:
        print(f"  AUROC:     {metrics['auroc']:.4f}")
    
    # Generate plots
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Confusion Matrix
    cm, cm_labels = plot_confusion_matrix_data(y_true, y_pred, class_names)
    fig = plot_confusion_matrix(
        cm, class_names,
        title='Confusion Matrix - Neural Classification',
        save_path=plots_dir / 'confusion_matrix.png'
    )
    print(f"\nSaved: confusion_matrix.png")
    
    # 2. ROC Curves
    roc_data = compute_roc_data(y_true, y_prob, n_classes=3)
    fig = plot_roc_curves(
        roc_data, class_names,
        title='ROC Curves - Neural Classification',
        save_path=plots_dir / 'roc_curves.png'
    )
    print(f"Saved: roc_curves.png")
    
    # 3. Calibration
    fig = plot_calibration(
        y_true, y_prob,
        title='Calibration - Neural Classification',
        save_path=plots_dir / 'calibration.png'
    )
    print(f"Saved: calibration.png")
    
    # 4. Sample spectrogram
    sample_spec = test_data['spectrograms'][0, 0]  # First sample, first channel
    fig = plot_spectrogram(
        sample_spec,
        title='Sample Spectrogram',
        save_path=plots_dir / 'sample_spectrogram.png'
    )
    print(f"Saved: sample_spectrogram.png")
    
    # 5. Connectivity graph
    sample_graph = test_data['graphs'][0]
    n_nodes = sample_graph.num_nodes
    adj_matrix = np.zeros((n_nodes, n_nodes))
    edge_idx = sample_graph.edge_index.cpu().numpy()
    edge_attr = sample_graph.edge_attr.cpu().numpy().flatten()
    for i in range(edge_idx.shape[1]):
        adj_matrix[edge_idx[0, i], edge_idx[1, i]] = edge_attr[i]
    
    fig = plot_connectivity_graph(
        adj_matrix,
        edge_threshold=0.3,
        title='Brain Connectivity Graph',
        save_path=plots_dir / 'connectivity_graph.png'
    )
    print(f"Saved: connectivity_graph.png")
    
    # 6. Grad-CAM (for CNN models)
    if args.model in ['cnn', 'hybrid']:
        try:
            target_model = model.cnn if args.model == 'hybrid' else model
            grad_cam = GradCAM(target_model)
            
            sample_input = torch.FloatTensor(test_data['spectrograms'][0:1]).to(device)
            cam, pred_class, conf = grad_cam(sample_input)
            
            fig = plot_saliency_map(
                test_data['spectrograms'][0, 0],
                cam,
                title=f'Grad-CAM (Pred: {class_names[pred_class]}, Conf: {conf:.2f})',
                save_path=plots_dir / 'grad_cam.png'
            )
            print(f"Saved: grad_cam.png")
        except Exception as e:
            print(f"Grad-CAM skipped: {e}")
    
    # 7. Band importance (mock for demo)
    band_importance = {
        'delta': 0.15,
        'theta': 0.25,
        'alpha': 0.35,
        'beta': 0.40,
        'gamma': 0.52
    }
    fig = plot_band_importance(
        band_importance,
        title='Frequency Band Importance',
        save_path=plots_dir / 'band_importance.png'
    )
    print(f"Saved: band_importance.png")
    
    return metrics


def main():
    """Main pipeline execution."""
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Configuration
    config = get_config()
    config.device = device
    
    # Output directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Force unique output directory per run
    project_root = Path(args.output_dir)
    run_dir = project_root / f"run_{timestamp}"
    
    output_dir = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    
    print(f"\n📂 Output Directory: {output_dir.absolute()}")
    
    # Load or generate data
    if args.demo or args.data is None:
        signals, labels, subject_ids = prepare_demo_data(config, output_dir)
    else:
        from preprocessing import load_neural_data
        signals, labels, subject_ids = load_neural_data(args.data)
        print(f"Loaded {len(signals)} samples from {args.data}")
    
    # Split data
    # Split data
    try:
        splits = subject_stratified_split(
            signals, labels, subject_ids,
            test_size=0.2, val_size=0.1, seed=config.seed
        )
    except Exception as e:
        print(f"\n! Stratified split failed: {e}")
        print("  Falling back to simple random split (WARNING: potential data leakage)")
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(signals))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=config.seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=config.seed)
        
        splits = {
            'train': (signals[train_idx], labels[train_idx], subject_ids[train_idx]),
            'val': (signals[val_idx], labels[val_idx], subject_ids[val_idx]),
            'test': (signals[test_idx], labels[test_idx], subject_ids[test_idx])
        }
    
    print(f"\nData splits:")
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val:   {len(splits['val'][0])} samples")
    print(f"  Test:  {len(splits['test'][0])} samples")
    
    # Extract features for each split
    train_specs, train_graphs = compute_features(splits['train'][0], config, device)
    val_specs, val_graphs = compute_features(splits['val'][0], config, device)
    test_specs, test_graphs = compute_features(splits['test'][0], config, device)
    
    train_data = {'spectrograms': train_specs, 'graphs': train_graphs, 'labels': splits['train'][1]}
    val_data = {'spectrograms': val_specs, 'graphs': val_graphs, 'labels': splits['val'][1]}
    test_data = {'spectrograms': test_specs, 'graphs': test_graphs, 'labels': splits['test'][1]}
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    
    if args.model == 'cnn':
        model = SpectrogramCNN(
            in_channels=1,
            n_classes=config.data.n_classes,
            base_filters=config.cnn.base_filters,
            embedding_dim=config.cnn.embedding_dim,
            dropout=config.cnn.dropout_rate
        )
    elif args.model == 'gnn':
        model = ConnectivityGNN(
            node_feature_dim=config.gnn.node_feature_dim,
            hidden_dim=config.gnn.hidden_dim,
            n_classes=config.data.n_classes,
            embedding_dim=config.gnn.embedding_dim,
            dropout=config.gnn.dropout_rate
        )
    else:  # hybrid
        model = HybridClassifier(
            cnn_config={
                'in_channels': 1,
                'base_filters': config.cnn.base_filters,
                'embedding_dim': config.cnn.embedding_dim
            },
            gnn_config={
                'node_feature_dim': config.gnn.node_feature_dim,
                'hidden_dim': config.gnn.hidden_dim,
                'embedding_dim': config.gnn.embedding_dim
            },
            n_classes=config.data.n_classes,
            fusion_type=config.hybrid.fusion_type
        )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    
    # Train
    history, _, _, _ = train_model(model, train_data, val_data, config, args, output_dir)
    
    # Plot training history
    fig = plot_training_history(
        history,
        title='Training History',
        save_path=output_dir / 'plots' / 'training_history.png'
    )
    print(f"\nSaved: training_history.png")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    model = model.to(device)
    
    metrics = evaluate_and_visualize(model, test_data, config, args, output_dir)
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"  - Best model: best_model.pt")
    print(f"  - Metrics: metrics.json")
    print(f"  - Plots: plots/")
    
    return metrics


if __name__ == '__main__':
    main()
