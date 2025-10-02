"""Lightweight tabular dataset helpers for PyTorch training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = ["TabularTensorDataset", "make_loaders"]


@dataclass
class TabularTensorDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Tensor-backed dataset for tabular features and integer labels.

    Ensures feature tensors use ``float32`` precision and targets are stored as
    ``int64`` (``torch.long``) for compatibility with common PyTorch losses.

    Args:
        X: Feature matrix shaped ``(n_samples, n_features)``.
        y: Target vector shaped ``(n_samples,)`` with integer class labels.

    Example:
        >>> X = torch.randn(10, 5)
        >>> y = torch.randint(0, 3, (10,))
        >>> dataset = TabularTensorDataset(X, y)
        >>> len(dataset)
        10
        >>> sample_X, sample_y = dataset[0]
        >>> sample_X.dtype, sample_y.dtype
        (torch.float32, torch.int64)

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "feat_1": [0.1, 0.5, -0.2],
        ...     "feat_2": [1.2, -0.7, 0.3],
        ...     "label": [0, 1, 0],
        ... })
        >>> dataset = TabularTensorDataset(
        ...     torch.tensor(df[["feat_1", "feat_2"]].values),
        ...     torch.tensor(df["label"].values),
        ... )
        >>> len(dataset)
        3
    """

    features: Tensor
    targets: Tensor

    def __post_init__(self) -> None:
        self.features = _to_float_tensor(self.features)
        self.targets = _to_long_tensor(self.targets)
        if self.features.size(0) != self.targets.size(0):
            raise ValueError("Features and targets must have the same number of samples.")

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]


def make_loaders(
    X_train: Iterable,
    y_train: Iterable,
    X_val: Iterable,
    y_val: Iterable,
    *,
    batch_size: int = 1024,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Construct train/validation dataloaders from array-like inputs.

    Args:
        X_train: Training features (tensor, numpy array, or sequence).
        y_train: Training targets with integer labels.
        X_val: Validation features.
        y_val: Validation targets.
        batch_size: Batch size to use for both loaders.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the training dataset each epoch.

    Returns:
        Tuple ``(train_loader, val_loader)`` ready for model training.

    Example:
        >>> train_loader, val_loader = make_loaders(
        ...     torch.randn(20, 4),
        ...     torch.randint(0, 2, (20,)),
        ...     torch.randn(8, 4),
        ...     torch.randint(0, 2, (8,)),
        ...     batch_size=8,
        ... )
        >>> next(iter(train_loader))[0].shape
        torch.Size([8, 4])

        >>> import pandas as pd
        >>> train_df = pd.DataFrame({
        ...     "f1": [0.1, 0.2, 0.3],
        ...     "f2": [1.0, 0.9, 1.1],
        ...     "label": [0, 1, 0],
        ... })
        >>> val_df = pd.DataFrame({
        ...     "f1": [-0.1, -0.2],
        ...     "f2": [0.8, 0.7],
        ...     "label": [1, 0],
        ... })
        >>> loaders = make_loaders(
        ...     train_df[["f1", "f2"]].values,
        ...     train_df["label"].values,
        ...     val_df[["f1", "f2"]].values,
        ...     val_df["label"].values,
        ...     batch_size=2,
        ... )
        >>> len(loaders[0])
        2
    """

    train_ds = TabularTensorDataset(_ensure_tensor(X_train), _ensure_tensor(y_train))
    val_ds = TabularTensorDataset(_ensure_tensor(X_val), _ensure_tensor(y_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def _ensure_tensor(data: Iterable) -> Tensor:
    if isinstance(data, Tensor):
        return data
    return torch.as_tensor(data)


def _to_float_tensor(tensor: Tensor) -> Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim < 2:
        raise ValueError("Feature tensor must be at least 2-D (batch, features).")
    return tensor


def _to_long_tensor(tensor: Tensor) -> Tensor:
    tensor = tensor.to(dtype=torch.long)
    if tensor.ndim != 1:
        tensor = tensor.view(-1)
    return tensor


if __name__ == "__main__":
    torch.manual_seed(0)
    X_tr = torch.randn(16, 4)
    y_tr = torch.randint(0, 3, (16,))
    X_va = torch.randn(8, 4)
    y_va = torch.randint(0, 3, (8,))

    train_loader, val_loader = make_loaders(X_tr, y_tr, X_va, y_va, batch_size=4)

    print("Train batches:")
    for xb, yb in train_loader:
        print(xb.shape, yb.shape, yb.dtype)

    print("Validation batches:")
    for xb, yb in val_loader:
        print(xb.shape, yb.shape, yb.dtype)
