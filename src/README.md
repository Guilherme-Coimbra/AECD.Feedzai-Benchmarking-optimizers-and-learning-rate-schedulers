# AECD Benchmarking Codebase Guide

This directory contains the building blocks for running tabular experiments in the notebooks. The modules are designed to stay lightweight (PyTorch-only) and composable, so you can mix-and-match models, optimizers, training loops, and visualizations.

## Models (`src/models`)
- **`FeedforwardNetwork`** (`src/models/ffn.py`): Configurable MLP for tabular data.
  ```python
  from src.models.ffn import FeedforwardNetwork

  model = FeedforwardNetwork(
      n_classes=3,
      n_features=20,
      hidden_layers=[
          {"units": 128, "activation": "relu", "dropout": 0.1},
          {"units": 64, "activation": "gelu", "normalization": "layernorm"},
          32,  # inherits default activation/dropout
      ],
      default_dropout=0.05,
  )
  ```

## Optimization (`src/optim`)
- **`create_optimizer`** & **`create_scheduler`** (`src/optim/factory.py`): Build optimizers/schedulers from small config dictionaries.
  ```python
  from src.optim.factory import create_optimizer, create_scheduler

  optimizer = create_optimizer(model, {"name": "adamw", "lr": 5e-4, "weight_decay": 1e-2})
  scheduler = create_scheduler(optimizer, {"name": "cosine", "t_max": 50})
  ```

## Training (`src/training`)
- **Metrics & Losses** (`src/training/metrics.py`): Conversions, accuracy/precision/recall/F1, and `get_loss_fn` (supports `binary`, `binary_weighted`, `multiclass`).
  ```python
  from src.training.metrics import get_loss_fn, accuracy

  loss_fn = get_loss_fn("binary_weighted", pos_weight=2.0)
  acc = accuracy(preds, targets)
  ```
- **Engine** (`src/training/engine.py`): Epoch-level training & evaluation loops with pluggable metrics.
  ```python
  from src.training.engine import train_one_epoch, evaluate

  history = train_one_epoch(model, train_loader, optimizer, loss_fn, device, {"accuracy": accuracy})
  val_metrics = evaluate(model, val_loader, loss_fn, device, {"accuracy": accuracy})
  ```

## Data (`src/data`)
- **`TabularTensorDataset`** & **`make_loaders`** (`src/data/datasets.py`): Minimal utilities for preparing PyTorch loaders from tensors/NumPy/pandas arrays.
  ```python
  from src.data.datasets import make_loaders

  train_loader, val_loader = make_loaders(
      X_train, y_train,
      X_val, y_val,
      batch_size=512,
  )
  ```

## Visualisation (`src/viz`)
- **Plot helpers** (`src/viz/plots.py`): Matplotlib utilities for history dictionaries (`plot_loss`, `plot_metric`, `plot_many_metrics`).
  ```python
  from src.viz.plots import plot_loss, plot_many_metrics

  fig_loss = plot_loss(history)
  figs = plot_many_metrics(history, ["accuracy", "f1"], title_prefix="Validation ")
  ```

---
**Tips**
- All modules are import-safe inside notebooks (`sys.path` already includes `src`).
- Each file has an executable `python src/...` smoke test if you want a quick sanity check.
- Extend functionality by following the TODO comments sprinkled throughout the modules.
