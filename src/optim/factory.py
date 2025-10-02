"""Utility helpers for constructing optimizers and schedulers from configs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler

__all__ = ["create_optimizer", "create_scheduler"]

_DEFAULT_BETAS: Tuple[float, float] = (0.9, 0.999)


def _get_parameters(model: Any) -> Iterable[torch.nn.Parameter]:
    """Extract model parameters in a way compatible with both nn.Modules and parameter groups."""

    if isinstance(model, torch.nn.Module):
        return model.parameters()
    if isinstance(model, Iterable):
        return model
    raise TypeError(
        "`model` must be an nn.Module or an iterable of parameters; "
        f"received {type(model)!r}."
    )


def create_optimizer(model: Any, cfg: Optional[Mapping[str, Any]] = None) -> Optimizer:
    """Instantiate an optimizer given a configuration mapping.

    Args:
        model: Either a :class:`torch.nn.Module` instance or an iterable of parameters.
        cfg: Mapping with optimizer hyper-parameters. Recognized keys are ``name``
            (default ``"adam"``; one of ``sgd``, ``adam``, ``adamw``, ``muon``),
            ``lr`` (default ``1e-3``), ``weight_decay`` (default ``0``), ``momentum``
            (SGD default ``0.9``), ``betas`` (Adam/AdamW default ``(0.9, 0.999)``),
            and ``nesterov`` (SGD default ``False``).

    Returns:
        A configured :class:`torch.optim.Optimizer` instance.

    Raises:
        ValueError: If the optimizer name is unknown.
        NotImplementedError: If the ``muon`` optimizer is requested.

    Example:
        >>> model = torch.nn.Linear(8, 4)
        >>> optimizer = create_optimizer(model, {"name": "sgd", "lr": 0.01})
        >>> isinstance(optimizer, torch.optim.SGD)
        True
    """

    cfg = dict(cfg or {})
    name = cfg.pop("name", "adam").lower()
    lr = cfg.pop("lr", 1e-3)
    weight_decay = cfg.pop("weight_decay", 0.0)

    params = _get_parameters(model)

    if name == "sgd":
        momentum = cfg.pop("momentum", 0.9)
        nesterov = cfg.pop("nesterov", False)
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **cfg,
        )
    elif name == "adam":
        betas = cfg.pop("betas", _DEFAULT_BETAS)
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=_coerce_betas(betas),
            weight_decay=weight_decay,
            **cfg,
        )
    elif name == "adamw":
        betas = cfg.pop("betas", _DEFAULT_BETAS)
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=_coerce_betas(betas),
            weight_decay=weight_decay,
            **cfg,
        )
    elif name == "muon":
        # TODO: Implement Muon optimizer wiring once available in the project.
        raise NotImplementedError("Muon optimizer support is not yet implemented.")
    else:
        raise ValueError(f"Unsupported optimizer '{name}'. Supported: sgd, adam, adamw, muon.")

    return optimizer


def _coerce_betas(value: Union[Tuple[float, float], Iterable[float]]) -> Tuple[float, float]:
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"`betas` must have exactly two floats; received {values}.")
    return values  # type: ignore[return-value]


def create_scheduler(
    optimizer: Optimizer,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[_LRScheduler]:
    """Instantiate a learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        cfg: Mapping that optionally contains a ``scheduler`` mapping, or is itself
            the scheduler mapping. Example::

            {"scheduler": {"name": "cosine", "t_max": 50}}

            {"name": "step", "step_size": 10, "gamma": 0.1}

    Returns:
        A :class:`torch.optim.lr_scheduler._LRScheduler` instance or ``None`` when
        no scheduler name is supplied.

    Raises:
        ValueError: If an unknown scheduler name is provided.

    Example:
        >>> model = torch.nn.Linear(8, 4)
        >>> optimizer = create_optimizer(model, {"name": "adam", "lr": 1e-3})
        >>> scheduler = create_scheduler(optimizer, {"name": "step", "step_size": 5})
        >>> isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        True
    """

    if cfg is None:
        return None

    scheduler_cfg: Optional[Mapping[str, Any]] = None
    if isinstance(cfg, Mapping):
        scheduler_cfg = cfg.get("scheduler")
        if scheduler_cfg is None and "name" in cfg:
            scheduler_cfg = cfg  # Assume cfg already describes the scheduler.
    if not scheduler_cfg:
        return None

    scheduler_cfg = dict(scheduler_cfg)
    name = scheduler_cfg.pop("name", None)

    if not name:
        return None

    name = name.lower()

    if name == "cosine":
        t_max = scheduler_cfg.pop("t_max", 50)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, **scheduler_cfg)
    elif name == "step":
        step_size = scheduler_cfg.pop("step_size", 10)
        gamma = scheduler_cfg.pop("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, **scheduler_cfg)
    else:
        raise ValueError(
            f"Unsupported scheduler '{name}'. Supported options are: cosine, step."
        )

    return scheduler


if __name__ == "__main__":
    torch.manual_seed(0)

    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )

    optimizer_cfg = {
        "name": "adamw",
        "lr": 5e-4,
        "weight_decay": 1e-2,
    }
    optimizer = create_optimizer(dummy_model, optimizer_cfg)
    scheduler = create_scheduler(
        optimizer,
        {"name": "cosine", "t_max": 25},
    )

    dummy_batch = torch.randn(3, 10)
    output = dummy_model(dummy_batch)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if scheduler is not None:
        scheduler.step()

    print("Optimizer type:", type(optimizer).__name__)
    print("Scheduler type:", type(scheduler).__name__ if scheduler else None)
    print("Loss value:", loss.item())
