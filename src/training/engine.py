"""Training and evaluation loops with pluggable metrics."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from src.training.metrics import logits_to_preds, accuracy

__all__ = ["train_one_epoch", "evaluate"]

MetricMap = Optional[Mapping[str, Callable[[Tensor, Tensor], float]]]


def _infer_task(logits: Tensor) -> str:
    if logits.ndim == 1:
        return "binary"
    if logits.ndim == 2 and logits.size(-1) == 1:
        return "binary"
    return "multiclass"


def _ensure_non_empty(dataloader: Iterable) -> Iterable:
    try:
        length = len(dataloader)  # type: ignore[arg-type]
    except TypeError:
        length = None

    if length is not None:
        if length == 0:
            raise ValueError(
                "Dataloader is empty; cannot proceed with training/evaluation."
            )
        return dataloader

    iterator = iter(dataloader)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError(
            "Dataloader is empty; cannot proceed with training/evaluation."
        ) from exc
    return chain([first], iterator)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    metric_fns: MetricMap = None,
) -> Dict[str, float]:
    """Run a single training pass over ``dataloader``.

    Args:
        model: Neural network to optimize.
        dataloader: Iterable yielding ``(inputs, targets)`` batches.
        optimizer: Optimizer instance with parameters from ``model``.
        loss_fn: Loss function that consumes logits and targets.
        device: Device on which computation is performed.
        metric_fns: Optional mapping from metric name to callable operating on
            concatenated predictions and targets (both on CPU).

    Returns:
        Dictionary with the average loss under the key ``"loss"`` and any
        additional metric outputs keyed by the provided mapping.

    Example:
        >>> model = nn.Linear(4, 2)
        >>> dataset = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> stats = train_one_epoch(
        ...     model,
        ...     loader,
        ...     optimizer,
        ...     nn.CrossEntropyLoss(),
        ...     torch.device("cpu"),
        ...     {"accuracy": accuracy},
        ... )
        >>> "loss" in stats and "accuracy" in stats
        True
    """

    data_iter = _ensure_non_empty(dataloader)
    model.train()

    total_loss = 0.0
    total_samples = 0
    preds_list = []
    targets_list = []

    for batch in data_iter:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        task = _infer_task(logits)
        preds, _ = logits_to_preds(logits.detach(), task=task)
        preds_list.append(preds.cpu())
        targets_list.append(targets.detach().cpu())

    return _finalize_metrics(total_loss, total_samples, preds_list, targets_list, metric_fns)


def evaluate(
    model: nn.Module,
    dataloader: Iterable,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    metric_fns: MetricMap = None,
) -> Dict[str, float]:
    """Evaluate ``model`` on ``dataloader`` without gradient updates.

    Args:
        model: Neural network to evaluate.
        dataloader: Iterable yielding ``(inputs, targets)`` batches.
        loss_fn: Loss function matching the model's output/target format.
        device: Device on which evaluation is performed.
        metric_fns: Optional mapping from metric name to callable operating on
            concatenated predictions and targets (both on CPU).

    Returns:
        Dictionary containing ``"loss"`` and any requested metrics.

    Example:
        >>> model = nn.Linear(4, 2)
        >>> dataset = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        >>> stats = evaluate(
        ...     model,
        ...     loader,
        ...     nn.CrossEntropyLoss(),
        ...     torch.device("cpu"),
        ...     {"accuracy": accuracy},
        ... )
        >>> "loss" in stats and "accuracy" in stats
        True
    """

    data_iter = _ensure_non_empty(dataloader)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in data_iter:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            task = _infer_task(logits)
            preds, _ = logits_to_preds(logits, task=task)
            preds_list.append(preds.cpu())
            targets_list.append(targets.cpu())

    return _finalize_metrics(total_loss, total_samples, preds_list, targets_list, metric_fns)


def _finalize_metrics(
    total_loss: float,
    total_samples: int,
    preds_list: Iterable[Tensor],
    targets_list: Iterable[Tensor],
    metric_fns: MetricMap,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["loss"] = total_loss / max(total_samples, 1)

    if metric_fns:
        preds_all = torch.cat(list(preds_list)) if preds_list else torch.tensor([])
        targets_all = torch.cat(list(targets_list)) if targets_list else torch.tensor([])
        for name, fn in metric_fns.items():
            metrics[name] = float(fn(preds_all, targets_all))

    return metrics


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)

    device = torch.device("cpu")
    train_features = torch.randn(32, 4)
    train_targets = torch.randint(0, 2, (32,), dtype=torch.long)
    val_features = torch.randn(16, 4)
    val_targets = torch.randint(0, 2, (16,), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_targets), batch_size=8)

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    metric_functions = {"accuracy": accuracy}

    train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, metric_functions)
    eval_metrics = evaluate(model, val_loader, loss_fn, device, metric_functions)

    print("Train metrics:", train_metrics)
    print("Eval metrics:", eval_metrics)
