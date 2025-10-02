"""Matplotlib helpers for visualising training histories."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

__all__ = ["plot_loss", "plot_metric", "plot_many_metrics"]


def plot_loss(history: Dict[str, Iterable[float]], title: str = "Loss"):
    """Plot training and validation loss curves if available.

    Args:
        history: Dictionary containing metric histories with optional ``"train_loss"``
            and ``"val_loss"`` keys.
        title: Title for the resulting figure.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure` containing the plot.

    Example:
        >>> history = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}
        >>> fig = plot_loss(history)
        >>> isinstance(fig, plt.Figure)
        True
    """

    fig, ax = plt.subplots()
    epochs = _infer_epochs(history)

    if "train_loss" in history:
        ax.plot(epochs[: len(history["train_loss"])], history["train_loss"], label="Train")
    if "val_loss" in history:
        ax.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="Validation")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric(history: Dict[str, Iterable[float]], key: str, title: Optional[str] = None):
    """Plot a metric for both training and validation splits.

    Args:
        history: Dictionary with metric histories.
        key: Metric name without the ``train_/val_`` prefix (e.g., ``"accuracy"``).
        title: Optional custom title; defaults to ``key.capitalize()``.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure` generated for the metric.

    Example:
        >>> hist = {"train_accuracy": [0.6, 0.7], "val_accuracy": [0.55, 0.65]}
        >>> fig = plot_metric(hist, "accuracy", "Accuracy")
        >>> isinstance(fig, plt.Figure)
        True
    """

    fig, ax = plt.subplots()
    epochs = _infer_epochs(history)
    title = title or key.capitalize()

    train_key = f"train_{key}"
    val_key = f"val_{key}"

    if train_key in history:
        ax.plot(epochs[: len(history[train_key])], history[train_key], label="Train")
    if val_key in history:
        ax.plot(epochs[: len(history[val_key])], history[val_key], label="Validation")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(key.replace("_", " ").title())
    ax.grid(True, linestyle="--", alpha=0.5)
    if train_key in history or val_key in history:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_many_metrics(history: Dict[str, Iterable[float]], keys: List[str], title_prefix: str = "") -> List:
    """Plot multiple metrics and return the created figures.

    Args:
        history: Dictionary containing metric histories.
        keys: List of metric names (without ``train_/val_`` prefixes) to plot.
        title_prefix: Optional prefix prepended to each generated figure title.

    Returns:
        A list of Matplotlib figures, one per metric.

    Example:
        >>> hist = {"train_f1": [0.5, 0.6], "val_f1": [0.45, 0.55]}
        >>> figs = plot_many_metrics(hist, ["f1"], title_prefix="Val: ")
        >>> len(figs)
        1
    """

    figures = []
    for key in keys:
        figure = plot_metric(history, key, f"{title_prefix}{key}")
        figures.append(figure)
    return figures


def _infer_epochs(history: Dict[str, Iterable[float]]):
    longest = max((len(values) for values in history.values()), default=0)
    return list(range(1, longest + 1))


if __name__ == "__main__":
    sample_history = {
        "train_loss": [1.0, 0.8, 0.6, 0.5],
        "val_loss": [1.1, 0.9, 0.85, 0.8],
        "train_accuracy": [0.5, 0.65, 0.7, 0.75],
        "val_accuracy": [0.48, 0.6, 0.62, 0.68],
    }

    fig_loss = plot_loss(sample_history)
    fig_metric = plot_metric(sample_history, "accuracy")
    figs = plot_many_metrics(sample_history, ["loss", "accuracy"], title_prefix="History: ")

    print("Generated figures:", len([fig_loss, fig_metric] + figs))
