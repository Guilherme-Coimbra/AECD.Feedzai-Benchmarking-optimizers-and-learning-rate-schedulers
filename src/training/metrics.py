"""Minimal classification metrics and loss helpers for training loops."""

# TODO: Extend loss factory with focal loss or other reweighting schemes.
# TODO: Add ranking-style metrics (e.g., precision@k, recall@k) for ranking experiments.

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "get_loss_fn",
    "weighted_bce_with_logits",
    "logits_to_preds",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    # Curves & ranking metrics (binary)
    "roc_curve_binary",
    "roc_auc_binary",
    "precision_at_k",
    "recall_at_k",
    "precision_recall_at_k",
]


def get_loss_fn(task: str, **kwargs: Any) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a standard PyTorch loss function for the given task.

    Args:
        task: ``"binary"`` for standard BCE-with-logits, ``"binary_weighted"`` for
            the weighted variant, or ``"multiclass"`` for multi-class cross-entropy.
        **kwargs: Optional parameters forwarded to the underlying loss constructor,
            such as ``pos_weight`` for binary tasks or ``weight`` for multi-class
            tasks.

    Returns:
        A callable loss function compatible with logits and integer targets.

    Raises:
        ValueError: If the task is not supported.

    Example:
        >>> loss_fn = get_loss_fn("binary")
        >>> isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
        True
        >>> weighted = get_loss_fn("binary_weighted", pos_weight=2.0)
        >>> isinstance(weighted, torch.nn.BCEWithLogitsLoss)
        True
    """

    task = task.lower()
    if task == "binary":
        return torch.nn.BCEWithLogitsLoss(**_filter_loss_kwargs(kwargs, "binary"))
    if task == "binary_weighted":
        filtered = _filter_loss_kwargs(kwargs, "binary")
        if "pos_weight" not in filtered:
            raise ValueError(
                "`pos_weight` must be provided when task='binary_weighted'."
            )
        return torch.nn.BCEWithLogitsLoss(**filtered)
    if task == "multiclass":
        return torch.nn.CrossEntropyLoss(**_filter_loss_kwargs(kwargs, "multiclass"))
    raise ValueError(
        f"Unsupported task '{task}'. Choose 'binary', 'binary_weighted', or 'multiclass'."
    )


def weighted_bce_with_logits(
    pos_weight: Union[float, Sequence[float], Tensor],
    *,
    weight: Optional[Union[float, Sequence[float], Tensor]] = None,
    reduction: str = "mean",
) -> torch.nn.BCEWithLogitsLoss:
    """Factory for Weighted BCE-with-logits loss.

    Args:
        pos_weight: Positive class weight (``w_p``) applied to the ``y = 1`` term.
            Larger values emphasize recall for the positive class.
        weight: Optional rescaling weight (``w_n``) applied per-example or per-class
            and multiplied with both positive and negative terms.
        reduction: Reduction to apply (``"mean"``, ``"sum"``, or ``"none"``).

    Returns:
        A ready-to-use :class:`torch.nn.BCEWithLogitsLoss` instance.

    Example:
        >>> loss_fn = weighted_bce_with_logits(pos_weight=2.0)
        >>> isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
        True

    Notes:
        The weighted BCE objective for a single sample can be written as::

            WBCE(y, y_hat) = -[w_p * y * log(y_hat) + w_n * (1 - y) * log(1 - y_hat)]

        where ``y_hat = sigmoid(logits)`` is the predicted probability of the
        positive class, ``y`` is the binary target, ``w_p`` = ``pos_weight`` and
        ``w_n`` defaults to ``1`` (or to ``weight`` when provided).
    """

    pos_weight_tensor = _to_tensor(pos_weight)
    weight_tensor = None if weight is None else _to_tensor(weight)
    return torch.nn.BCEWithLogitsLoss(
        weight=weight_tensor,
        pos_weight=pos_weight_tensor,
        reduction=reduction,
    )


def logits_to_preds(logits: Tensor, task: str) -> Tuple[Tensor, Tensor]:
    """Convert raw logits to predicted class indices and probabilities.

    Args:
        logits: Model outputs. Shape ``(batch,)`` or ``(batch, 1)`` for binary tasks
            and ``(batch, num_classes)`` for multi-class tasks.
        task: Either ``"binary"`` or ``"multiclass"``.

    Returns:
        Tuple of ``(pred_classes, probabilities)`` where classes are ``LongTensor``
        and probabilities lie in ``[0, 1]``.

    Example:
        >>> logits = torch.tensor([[1.0, -0.5], [-0.2, 0.1]])
        >>> preds, probs = logits_to_preds(logits, "multiclass")
        >>> preds.tolist()
        [0, 1]
    """

    task = task.lower()
    if task == "binary":
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (logits >= 0).long()
        return preds, probs

    if task == "multiclass":
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return preds.long(), probs

    raise ValueError(f"Unsupported task '{task}'. Choose 'binary' or 'multiclass'.")


def accuracy(preds: Tensor, targets: Tensor) -> float:
    """Compute the proportion of correct predictions."""

    preds, targets = _prepare(preds, targets)
    return torch.mean((preds == targets).float()).item()


def precision(preds: Tensor, targets: Tensor) -> float:
    """Compute macro-averaged precision for single-label classification."""

    return _macro_stat(preds, targets, kind="precision")


def recall(preds: Tensor, targets: Tensor) -> float:
    """Compute macro-averaged recall for single-label classification."""

    return _macro_stat(preds, targets, kind="recall")


def f1_score(preds: Tensor, targets: Tensor) -> float:
    """Compute macro-averaged F1 score for single-label classification."""

    return _macro_stat(preds, targets, kind="f1")


def _prepare(preds: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    preds = preds.view(-1)
    targets = targets.view(-1)
    if preds.numel() != targets.numel():
        raise ValueError(
            "Predictions and targets must contain the same number of elements."
        )
    return preds.to(torch.long), targets.to(torch.long)


def _macro_stat(preds: Tensor, targets: Tensor, kind: str) -> float:
    preds, targets = _prepare(preds, targets)
    classes = torch.unique(torch.cat([preds, targets])).to(torch.long)
    scores = []

    for cls in classes:
        tp = torch.sum((preds == cls) & (targets == cls)).float()
        fp = torch.sum((preds == cls) & (targets != cls)).float()
        fn = torch.sum((preds != cls) & (targets == cls)).float()

        if kind == "precision":
            denom = tp + fp
            score = tp / denom if denom > 0 else tp.new_tensor(0.0)
        elif kind == "recall":
            denom = tp + fn
            score = tp / denom if denom > 0 else tp.new_tensor(0.0)
        elif kind == "f1":
            prec = tp / (tp + fp) if tp + fp > 0 else tp.new_tensor(0.0)
            rec = tp / (tp + fn) if tp + fn > 0 else tp.new_tensor(0.0)
            denom = prec + rec
            score = (2 * prec * rec / denom) if denom > 0 else tp.new_tensor(0.0)
        else:
            raise ValueError(f"Unknown metric kind '{kind}'.")

        scores.append(score)

    if not scores:
        return 0.0
    return torch.stack(scores).mean().item()


# -------------------------
# ROC curve (binary only)
# -------------------------

def roc_curve_binary(
    targets: Tensor,
    scores: Tensor,
    *,
    from_logits: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute ROC curve points (FPR, TPR, thresholds) for binary targets.

    Args:
        targets: 1D tensor with binary ground-truth labels {0,1}.
        scores: 1D tensor with predicted scores; probabilities in [0,1] or raw logits.
        from_logits: If True, applies sigmoid to ``scores`` before computing the curve.

    Returns:
        Tuple ``(fpr, tpr, thresholds)`` as tensors sorted by descending threshold.

    Notes:
        - This function mirrors ``sklearn.metrics.roc_curve`` behavior closely.
        - Thresholds are returned from ``+inf`` down to the lowest unique score.
    """

    targets = targets.detach().view(-1).to(torch.long)
    scores = scores.detach().view(-1).to(torch.float)
    if targets.numel() == 0:
        raise ValueError("Empty inputs: targets and scores must be non-empty.")
    if targets.numel() != scores.numel():
        raise ValueError("targets and scores must have the same number of elements.")

    if from_logits:
        scores = torch.sigmoid(scores)

    # Sort by score descending
    order = torch.argsort(scores, descending=True)
    y_true = targets[order]
    y_score = scores[order]

    # Total positives/negatives
    P = torch.sum(y_true == 1).item()
    N = torch.sum(y_true == 0).item()
    if P == 0 or N == 0:
        # Degenerate case: cannot form a ROC curve
        fpr = torch.tensor([0.0, 1.0])
        tpr = torch.tensor([0.0, 1.0])
        thresholds = torch.tensor([float("inf"), float("-inf")])
        return fpr, tpr, thresholds

    # True/false positive cumulative counts as threshold moves downward
    y_true_float = y_true.to(torch.float)
    tps = torch.cumsum(y_true_float, dim=0)
    fps = torch.cumsum(1.0 - y_true_float, dim=0)

    # Identify indices where threshold changes (unique scores)
    # We include the last index to capture the final point
    diff = torch.diff(y_score, prepend=y_score[:1] - 1)  # force first as change
    change_idx = torch.nonzero(diff != 0, as_tuple=False).view(-1)
    change_idx = torch.cat([change_idx, torch.tensor([y_score.numel() - 1])]).unique()
    change_idx, _ = torch.sort(change_idx)

    # Compute FPR/TPR at these thresholds
    tpr = tps[change_idx] / max(P, 1)
    fpr = fps[change_idx] / max(N, 1)
    thresholds = y_score[change_idx]

    # Prepend (0,0) at threshold +inf to start the curve
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    thresholds = torch.cat([torch.tensor([float("inf")]), thresholds])

    return fpr, tpr, thresholds


def roc_auc_binary(fpr: Tensor, tpr: Tensor) -> float:
    """Area under the ROC curve using the trapezoidal rule.

    Inputs should come from :func:`roc_curve_binary`. ``fpr`` must be increasing.
    """

    if fpr.numel() < 2:
        return 0.0
    # Ensure sorted by FPR ascending
    order = torch.argsort(fpr)
    xf = fpr[order]
    yf = tpr[order]
    dx = torch.diff(xf)
    y_sum = yf[:-1] + yf[1:]
    auc = torch.sum(dx * y_sum * 0.5).item()
    return float(auc)


# --------------------------------
# Precision@k and Recall@k (binary)
# --------------------------------

def _normalize_k(k: Union[int, float], n: int) -> int:
    if isinstance(k, float):
        if not (0.0 < k <= 1.0):
            raise ValueError("Fractional k must satisfy 0 < k <= 1.")
        k = int(max(1, round(k * n)))
    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer or a fraction in (0,1].")
    return min(k, n)


def precision_recall_at_k(
    targets: Tensor,
    scores: Tensor,
    ks: Sequence[Union[int, float]],
    *,
    from_logits: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute precision@k and recall@k for a list of k.

    Args:
        targets: 1D binary labels {0,1}.
        scores: 1D scores (probabilities or logits).
        ks: Sequence of k values, each int or fraction in (0,1] for top-k%.
        from_logits: If True, applies sigmoid to ``scores``.

    Returns:
        Tuple of tensors ``(k_tensor, precision, recall)`` aligned by index.
    """

    targets = targets.detach().view(-1).to(torch.long)
    scores = scores.detach().view(-1).to(torch.float)
    if from_logits:
        scores = torch.sigmoid(scores)
    if targets.numel() != scores.numel():
        raise ValueError("targets and scores must have the same number of elements.")

    n = targets.numel()
    total_pos = float((targets == 1).sum().item())
    # Sort by scores descending
    order = torch.argsort(scores, descending=True)
    y_sorted = targets[order]
    cum_pos = torch.cumsum(y_sorted == 1, dim=0).to(torch.float)

    k_ints = torch.tensor([_normalize_k(k, n) for k in ks], dtype=torch.long)
    prec = torch.zeros_like(k_ints, dtype=torch.float)
    rec = torch.zeros_like(k_ints, dtype=torch.float)

    for i, k_val in enumerate(k_ints):
        tp_k = cum_pos[k_val.item() - 1]
        prec[i] = tp_k / float(k_val.item())
        rec[i] = tp_k / total_pos if total_pos > 0 else torch.tensor(0.0)

    return k_ints.to(torch.float), prec, rec


def precision_at_k(
    targets: Tensor,
    scores: Tensor,
    k: Union[int, float],
    *,
    from_logits: bool = False,
) -> float:
    ks, p, _ = precision_recall_at_k(targets, scores, [k], from_logits=from_logits)
    return float(p.item())


def recall_at_k(
    targets: Tensor,
    scores: Tensor,
    k: Union[int, float],
    *,
    from_logits: bool = False,
) -> float:
    ks, _, r = precision_recall_at_k(targets, scores, [k], from_logits=from_logits)
    return float(r.item())


def _filter_loss_kwargs(args: dict, task: str) -> dict:
    allowed = {
        "binary": {"weight", "size_average", "reduce", "reduction", "pos_weight"},
        "multiclass": {"weight", "size_average", "ignore_index", "reduce", "reduction"},
    }
    permitted = allowed.get(task, set())
    filtered = {}
    for key, value in args.items():
        if key not in permitted:
            continue
        if key in {"weight", "pos_weight"}:
            filtered[key] = _to_tensor(value)
        else:
            filtered[key] = value
    return filtered


def _to_tensor(value: Union[float, Sequence[float], Tensor]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (float, int)):
        return torch.tensor(float(value))
    if isinstance(value, Sequence):
        return torch.tensor(value, dtype=torch.float)
    raise TypeError(f"Unable to convert value of type {type(value)!r} to a tensor.")


if __name__ == "__main__":
    torch.manual_seed(0)

    logits = torch.randn(4, 3)
    preds, probs = logits_to_preds(logits, task="multiclass")
    targets = torch.tensor([0, 2, 1, 2])

    print("Loss fn:", get_loss_fn("multiclass").__class__.__name__)
    print("Accuracy:", accuracy(preds, targets))
    print("Precision:", precision(preds, targets))
    print("Recall:", recall(preds, targets))
    print("F1 score:", f1_score(preds, targets))

    wbce = weighted_bce_with_logits(pos_weight=2.5)
    wbce_factory = get_loss_fn("binary_weighted", pos_weight=2.5)
    binary_logits = torch.randn(6)
    binary_targets = torch.randint(0, 2, (6,), dtype=torch.float)
    print("Weighted BCE loss:", wbce(binary_logits, binary_targets).item())
    print(
        "Weighted BCE via factory:",
        wbce_factory(binary_logits, binary_targets).item(),
    )

    print("WBCE(y, y_hat) = -[w_p * y * log(y_hat) + w_n * (1 - y) * log(1 - y_hat)]")
    print("where y_hat = sigmoid(logits), w_p = pos_weight, and w_n = 1 (or `weight`).")
