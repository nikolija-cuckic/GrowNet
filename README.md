# GrowNet — Gradient Boosting Neural Networks

Implementation of the **GrowNet** architecture — a model based on the principle of gradient boosting over shallow neural networks (weak learners). The project includes training a GrowNet model, comparison with a Baseline MLP and XGBoost, and an ablation study analyzing the impact of key hyperparameters.

---

## Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Models](#running-the-models)
- [Hyperparameters (config.py)](#hyperparameters-configpy)
- [Logging and Results](#logging-and-results)

---

## About

GrowNet is a boosting framework that combines the strengths of gradient boosting (XGBoost-style) with the flexibility of neural networks. Instead of decision trees, it uses shallow MLPs as **weak learners**. At each boosting step (stage), a new weak learner is trained on the **pseudo-residual** — the negative gradient of the loss with respect to the current ensemble's prediction.

Optionally, a **Corrective Step (CS)** is applied: an end-to-end fine-tuning of the entire ensemble to correct accumulated error.

---

## Architecture

```
Input (features)
     │
     ▼
 WeakLearner_1 ──┐
 WeakLearner_2 ──┤  × shrinkage (η)   →   Sum of predictions
      ...        │
 WeakLearner_N ──┘
     │
     ▼
  Final prediction
```

- **WeakLearner**: Shallow MLP (`input → hidden → hidden → 1`) with ReLU activations
- **GrowNet**: A dynamic container (`nn.ModuleList`) that adds a new weak learner at each stage
- **Shrinkage (η)**: A scaling factor that controls the contribution of each weak learner (regularization)
- **Corrective Step**: Optional end-to-end fine-tuning applied every `CS_EVERY` stages

---

## Project Structure

```
GrowNet/
│
├── config.py                    # Central configuration (dataset, hyperparameters)
│
├── main_grownet.py              # Train a single GrowNet model
├── main_grownet_sweep.py        # Grid search sweep for GrowNet
├── main_baseline.py             # Train a Baseline MLP model
├── main_baseline_sweep.py       # Grid search sweep for Baseline MLP
├── main_xgboost.py              # Train an XGBoost model (for comparison)
│
├── ablation_study.ipynb         # Jupyter notebook for result analysis and visualization
│
├── models/
│   ├── weak_learner.py          # WeakLearner (shallow MLP)
│   ├── baseline_mlp.py          # Baseline MLP model
│   └── grownet.py               # GrowNet ensemble
│
├── training/
│   ├── base_trainer.py          # Base class with shared training logic
│   ├── baseline_trainer.py      # Trainer for Baseline MLP
│   └── grownet_trainer.py       # Trainer for GrowNet (boosting logic + CS)
│
├── utils/
│   ├── data_loader.py           # Data loading and normalization
│   ├── preprocess_data.py       # Raw CSV preprocessing
│   ├── logger.py                # Metric logging to CSV
│   └── plotting.py              # Training curve plots
│
├── data/
│   ├── raw/                     # Raw CSV files (added manually)
│   └── processed/               # Preprocessed and normalized data (auto-generated)
│
├── checkpoints/                 # Saved model weights (.pt files)
├── logs/                        # CSV experiment logs
└── plots/                       # Generated plots
```

---

## Supported Datasets

| Dataset | Task | Target Column | Notes |
|---|---|---|---|
| `california_housing` | Regression | `median_house_value` | Sklearn / Kaggle |
| `slice_localization` | Regression | `reference` | UCI Repository |
| `higgs` | Classification | `class_label` | Supports 100k and 1M subsets |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd GrowNet

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> **Note:** If an NVIDIA GPU is available, PyTorch will use it automatically. Otherwise, training runs on CPU.

---

## Data Preparation

Raw CSV files must be placed in `data/raw/` before preprocessing:

| Dataset | File |
|---|---|
| California Housing | `data/raw/housing.csv` |
| Slice Localization | `data/raw/slice_localization_data.csv` |
| HIGGS | `data/raw/HIGGS.csv` |

Run the preprocessing script:

```bash
python -m utils.preprocess_data
```

This generates normalized `.pt` files in `data/processed/<dataset_name>/`.

---

## Running the Models

### GrowNet

```bash
python main_grownet.py
```

### Baseline MLP

```bash
python main_baseline.py
```

### XGBoost (for comparison)

```bash
python main_xgboost.py
```

### Grid Search Sweep (GrowNet)

```bash
python main_grownet_sweep.py
```

The sweep runs all defined hyperparameter combinations and logs results to `logs/experiments.csv`.

### Ablation Study

Open `ablation_study.ipynb` in a Jupyter environment for interactive analysis and visualization of sweep results.

---

## Hyperparameters (config.py)

All key parameters are defined in `config.py` and can be changed before running training.

### Dataset

| Parameter | Default | Description |
|---|---|---|
| `BASE_DATASET_NAME` | `'slice_localization'` | Active dataset |
| `HIGGS_SIZE` | `'1M'` | HIGGS subset size (`100k` or `1M`) |
| `TEST_SIZE` | `0.2` | Fraction of data used for testing |
| `BATCH_SIZE` | `512` | Mini-batch size |

### GrowNet

| Parameter | Default | Description |
|---|---|---|
| `GROWNET_NUM_STAGES` | `10` | Number of boosting stages (weak learners) |
| `GROWNET_WEAK_HIDDEN_DIM` | `64` | Hidden dimension of each weak learner |
| `GROWNET_WEAK_LR` | `0.001` | Learning rate for weak learner training |
| `GROWNET_SHRINKAGE` | `0.1` | Shrinkage factor (boosting rate) |
| `GROWNET_USE_CS` | `True` | Whether to apply the Corrective Step |
| `GROWNET_CS_EPOCHS` | `1` | Number of Corrective Step epochs |
| `GROWNET_CS_EVERY` | `1` | Apply Corrective Step every N stages |

### Baseline MLP

| Parameter | Default | Description |
|---|---|---|
| `BASELINE_HIDDEN_DIM` | `64` | Hidden layer dimension |
| `BASELINE_LAYERS` | `2` | Number of hidden layers |
| `BASELINE_LEARNING_RATE` | `0.001` | Learning rate |
| `BASELINE_EPOCHS` | `100` | Number of training epochs |

### Early Stopping

| Parameter | Default | Description |
|---|---|---|
| `EARLY_STOPPING_PATIENCE` | `15` | Steps without improvement before stopping |
| `EARLY_STOPPING_MIN_DELTA` | `1e-6` | Minimum change considered as improvement |

---

## Logging and Results

- **`logs/`** — Per-epoch/stage CSV metric logs for each experiment
- **`logs/experiments.csv`** — Summary of all completed experiments (sweep)
- **`checkpoints/`** — Saved `.pt` model weights (best checkpoint)
- **`plots/`** — Training and test loss curves
- **`ablation_study.ipynb`** — Visualization and analysis of hyperparameter impact (L2 regularization, Corrective Step, etc.)

---

## Reference

- Huang, C. et al. (2020). *[GrowNet: Gradient Boosting Neural Networks](https://arxiv.org/abs/2002.07971)*. arXiv:2002.07971
