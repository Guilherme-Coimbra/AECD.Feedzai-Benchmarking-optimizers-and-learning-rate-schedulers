#!/usr/bin/env python

# TODO: Extend FeedforwardNetwork for richer notebook experiments:
# - Make dropout optional on the final hidden layer when regularization is not desired.
# - Expose hooks for extras such as residual connections or custom initialization.
# - Add convenience helpers (e.g., from_config factory, predict_proba) to simplify usage.

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Mapping, Optional, Sequence, Union

__all__ = ["FeedforwardNetwork"]


class FeedforwardNetwork(nn.Module):
    """Stacked feed-forward network tailored to small tabular datasets.

    Example:
        >>> model = FeedforwardNetwork(
        ...     n_classes=3,
        ...     n_features=10,
        ...     hidden_layers=[
        ...         {"units": 512, "activation": "relu", "dropout": 0.1},
        ...         {"units": 256, "activation": "tanh", "normalization": "layernorm"},
        ...         128,
        ...     ],
        ... )
        >>> batch = torch.randn(8, 10)
        >>> logits = model(batch)
        >>> logits.shape
        torch.Size([8, 3])
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        hidden_layers: Sequence[Union[int, Mapping[str, Any]]],
        *,
        default_activation: str = "relu",
        default_dropout: float = 0.0,
        default_normalization: Optional[str] = None,
    ) -> None:
        super().__init__()
        """Build a feed-forward classifier with configurable hidden blocks.

        Args:
            n_classes: Number of output classes (logits per sample).
            n_features: Number of input features each sample provides.
            hidden_layers: Sequence describing each hidden block. Elements can be
                integers (interpreted as output units) or mappings with keys
                ``units`` (required) and optional ``activation``, ``dropout``,
                ``normalization``.
            default_activation: Fallback activation name when a block omits it.
            default_dropout: Fallback dropout probability when a block omits it.
            default_normalization: Fallback normalization name when a block
                omits it. Supported normalizations: ``batchnorm``, ``layernorm``.

        Raises:
            ValueError: If any dimension parameter is non-positive, dropout is
                out of range, or the activation/normalization is unsupported.
        """

        if n_classes <= 0:
            raise ValueError("`n_classes` must be a positive integer.")
        if n_features <= 0:
            raise ValueError("`n_features` must be a positive integer.")
        if not hidden_layers:
            raise ValueError("`hidden_layers` must contain at least one block.")
        if not (0.0 <= default_dropout <= 1.0):
            raise ValueError("`default_dropout` must be in the interval [0, 1].")

        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        normalization_map = {
            "batchnorm": nn.BatchNorm1d,
            "layernorm": nn.LayerNorm,
        }

        self.hidden_layers = nn.ModuleList()
        in_features = n_features

        for index, spec in enumerate(hidden_layers):
            if isinstance(spec, Mapping):
                units = spec.get("units")
                activation_name = spec.get("activation", default_activation)
                dropout = spec.get("dropout", default_dropout)
                normalization_name = spec.get("normalization", default_normalization)
            else:
                units = spec
                activation_name = default_activation
                dropout = default_dropout
                normalization_name = default_normalization

            if units is None:
                raise ValueError(f"Hidden layer {index} missing required 'units'.")
            if not isinstance(units, int) or units <= 0:
                raise ValueError(f"Hidden layer {index} has invalid 'units': {units!r}.")
            if not (0.0 <= dropout <= 1.0):
                raise ValueError(
                    f"Hidden layer {index} has dropout outside [0, 1]: {dropout!r}."
                )

            try:
                activation_cls = activation_map[activation_name.lower()]
            except KeyError as exc:
                raise ValueError(
                    f"Hidden layer {index} has unsupported activation '{activation_name}'. "
                    "Supported activations are: relu, tanh, gelu."
                ) from exc

            if normalization_name is not None:
                try:
                    normalization_cls = normalization_map[normalization_name.lower()]
                except KeyError as exc:
                    raise ValueError(
                        f"Hidden layer {index} has unsupported normalization "
                        f"'{normalization_name}'. Supported options are: batchnorm, layernorm."
                    ) from exc
            else:
                normalization_cls = None

            modules = [nn.Linear(in_features, units)]
            if normalization_cls is not None:
                modules.append(normalization_cls(units))
            modules.append(activation_cls())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

            self.hidden_layers.append(nn.Sequential(*modules))
            in_features = units

        self.output_layer = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute class logits for a batch of tabular samples.

        Example:
            >>> model = FeedforwardNetwork(
            ...     n_classes=2,
            ...     n_features=6,
            ...     hidden_layers=[
            ...         {"units": 16, "activation": "gelu", "dropout": 0.2},
            ...         {"units": 8, "activation": "tanh"},
            ...     ],
            ... )
            >>> samples = torch.randn(4, 6)
            >>> model(samples).shape
            torch.Size([4, 2])
        """
        for block in self.hidden_layers:
            x = block(x)
        return self.output_layer(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    demo_model = FeedforwardNetwork(
        n_classes=3,
        n_features=10,
        hidden_layers=[
            {"units": 32, "activation": "relu", "dropout": 0.1},
            {"units": 16, "activation": "gelu", "normalization": "batchnorm"},
            8,
        ],
    )
    demo_batch = torch.randn(5, 10)
    demo_logits = demo_model(demo_batch)
    print("Demo logits shape:", demo_logits.shape)
