# Neural Activity Classification Pipeline

A comprehensive, research-grade machine learning pipeline for classifying neural activity data into **healthy**, **schizophrenia**, and **bipolar disorder** using PyTorch.

## Overview

This pipeline provides end-to-end tools for analyzing neural signals from:
- **Brain Organoids** (MEA recordings)
- **EEG** (Electroencephalography)
- **fMRI** (Functional MRI)

### Key Features

- **Hybrid CNN+GNN Architecture**: Combines spectral analysis with connectivity graph analysis
- **Comprehensive Feature Engineering**: STFT spectrograms, PSD, coherence matrices, band power
- **Explainability**: Grad-CAM, attention visualization, biomarker extraction
- **Clinical Translation**: Drug response simulation, modality adapters
- **Uncertainty Estimation**: Monte Carlo dropout, temperature scaling

## Project Structure

```
neural_classifier/
├── config/                 # Configuration dataclasses
│   └── config.py
├── preprocessing/          # Data loading, filtering, normalization
│   ├── data_loader.py
│   ├── filters.py
│   └── normalization.py
├── features/               # Feature extraction
│   ├── spectral.py        # STFT, PSD, band power
│   ├── coherence.py       # Connectivity metrics
│   └── graph_builder.py   # PyG graph construction
├── models/                 # Neural network architectures
│   ├── cnn_branch.py      # Spectrogram CNN
│   ├── gnn_branch.py      # Connectivity GNN
│   ├── hybrid_fusion.py   # Combined model
│   └── attention.py       # Attention mechanisms
├── training/               # Training infrastructure
│   ├── trainer.py         # Training loop, k-fold CV
│   ├── metrics.py         # Evaluation metrics
│   └── uncertainty.py     # Calibration, MC dropout
├── explainability/         # Model interpretation
│   ├── grad_cam.py        # CNN visualization
│   └── gnn_explain.py     # GNN visualization
├── visualization/          # Plotting utilities
│   └── plots.py
├── clinical/               # Clinical tools
│   ├── translation.py     # Modality adapters
│   └── drug_simulation.py # Drug effect simulation
├── main.py                 # Pipeline orchestration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Quick Start

### Installation

```bash
# Clone or download the project
cd neural_classifier

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Run with synthetic data
python main.py --demo

# This will:
# 1. Generate synthetic neural data for 3 classes
# 2. Train a hybrid CNN+GNN model
# 3. Generate evaluation plots in outputs/plots/
```

### Train on Your Data

```bash
# Basic usage
python main.py --data path/to/your/data.csv

# Full options
python main.py \
    --data data/eeg_signals.csv \
    --modality eeg \
    --model hybrid \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir results
```

## Data Format

Expected CSV format:

```csv
subject_id,segment_id,label,ch_0,ch_1,ch_2,...,ch_63
1,0,0,0.123,0.456,0.789,...
1,0,0,0.234,0.567,0.890,...
```

- `subject_id`: Unique subject identifier (for stratification)
- `segment_id`: Segment within subject
- `label`: 0=healthy, 1=schizophrenia, 2=bipolar
- `ch_X`: Channel voltage values (one row per timepoint)

##  Configuration

Edit `config/config.py` to customize:

```python
# Frequency bands
FrequencyBands:
    delta: (0.5, 4.0)
    theta: (4.0, 8.0)
    alpha: (8.0, 13.0)
    beta: (13.0, 30.0)
    gamma: (30.0, 100.0)

# Model settings
CNNConfig:
    base_filters: 32
    n_blocks: 4
    dropout_rate: 0.5

GNNConfig:
    hidden_dim: 64
    n_layers: 3
    gnn_type: 'gat'  # 'gcn', 'gat', 'sage'
```

## Model Architectures

### CNN Branch
- Processes time-frequency spectrograms
- Residual blocks with CBAM attention
- Global average pooling

### GNN Branch
- Processes brain connectivity graphs
- GAT layers with multi-head attention
- Attention-based graph pooling

### Hybrid Fusion
- Combines CNN and GNN embeddings
- Attention-weighted fusion
- Joint classification head

## Outputs

After training, find in `outputs/`:

```
outputs/
├── best_model.pt           # Trained model weights
├── metrics.json            # Evaluation metrics
└── plots/
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── calibration.png
    ├── training_history.png
    ├── sample_spectrogram.png
    ├── connectivity_graph.png
    ├── grad_cam.png
    └── band_importance.png
```

##  Clinical Tools

### Drug Response Simulation

```python
from clinical import DrugSimulator

simulator = DrugSimulator(sample_rate=256.0)

# Simulate antipsychotic effect
modified = simulator.apply_drug(signal, 'haloperidol', dose_factor=1.0)

# Available drugs: haloperidol, clozapine, lithium, valproate, fluoxetine, sertraline
```

### Modality Adaptation

```python
from clinical import EEGAdapter, OrganoidAdapter, fMRIAdapter

# Configure for EEG
adapter = EEGAdapter(sample_rate=256.0, n_channels=64)
config = adapter.get_config()
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{neural_classifier,
  title={Neural Activity Classification Pipeline},
  year={2024},
  description={ML pipeline for psychiatric disorder classification from neural signals}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please submit issues and pull requests.
