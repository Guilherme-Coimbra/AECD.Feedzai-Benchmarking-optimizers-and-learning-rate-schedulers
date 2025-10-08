#!/usr/bin/env python

"""
FeedforwardNetwork: fully-connected neural network for small tabular datasets
with optional normalization, dropout, and flexible residual (skip) connections.

Residual connections can link *any* two “taps” of the computation graph:
- `pre0` : before the first layer (input)
- `pre{i}` : before hidden layer i
- `post{i}` : after hidden layer i
- `pre_out` : before output layer
- `post_out` : after output layer (logits)

Each residual is defined as a mapping:
    {"from": ("pre"|"post", idx_from),
     "to":   ("pre"|"post", idx_to)}

If input/output dims differ, a Linear projection is created automatically.
All skips use additive combination (`dest = dest + proj(src)`).

Example
-------
>>> import torch
>>> from ffn import FeedforwardNetwork
>>> # Basic model
>>> model = FeedforwardNetwork(
...     n_classes=2,
...     n_features=8,
...     hidden_layers=[64, 32],
... )
>>> x = torch.randn(4, 8)
>>> model(x).shape
torch.Size([4, 2])
>>>
>>> # Model with residuals:
>>> residuals = [
...     {"from": ("pre", 1), "to": ("post", 1)},  # within layer 1
...     {"from": ("pre", 0), "to": ("post", 2)},  # input -> after layer 2
... ]
>>> model = FeedforwardNetwork(
...     n_classes=2,
...     n_features=8,
...     hidden_layers=[64, 64],
...     residual_specs=residuals,
... )
>>> y = model(torch.randn(4, 8))
>>> y.shape
torch.Size([4, 2])
"""

from __future__ import annotations
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, Union, List, Tuple


__all__ = ["FeedforwardNetwork"]


class FeedforwardNetwork(nn.Module):
    """Stacked feed-forward network with flexible residual connections.

    Args
    ----
    n_classes : int
        Number of output logits per sample.
    n_features : int
        Number of input features per sample.
    hidden_layers : sequence of int or mapping
        Each element can be an int (units) or dict with:
        {"units", "activation", "dropout", "normalization"}.
    default_activation : str, default="relu"
        Fallback activation when unspecified.
    default_dropout : float, default=0.0
        Fallback dropout probability.
    default_normalization : {"batchnorm","layernorm",None}
        Optional normalization type.
    residual_specs : list of dict, optional
        Arbitrary skip connections. Each element:
            {"from": ("pre"|"post", idx_from),
             "to": ("pre"|"post", idx_to)}

    Example
    -------
    >>> net = FeedforwardNetwork(
    ...     n_classes=3,
    ...     n_features=10,
    ...     hidden_layers=[
    ...         {"units": 64, "activation": "relu"},
    ...         {"units": 32, "activation": "tanh"},
    ...     ],
    ...     residual_specs=[{"from": ("pre", 1), "to": ("post", 1)}],
    ... )
    >>> out = net(torch.randn(5, 10))
    >>> out.shape
    torch.Size([5, 3])
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
        residual_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__()

        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        if not (0.0 <= default_dropout <= 1.0):
            raise ValueError("default_dropout must be in [0,1]")

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
        dims_in, dims_out = [], []

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

            if not isinstance(units, int) or units <= 0:
                raise ValueError(f"Layer {index}: invalid units {units}")

            try:
                activation_cls = activation_map[activation_name.lower()]
            except KeyError:
                raise ValueError(f"Unsupported activation {activation_name}")

            normalization_cls = None
            if normalization_name is not None:
                try:
                    normalization_cls = normalization_map[normalization_name.lower()]
                except KeyError:
                    raise ValueError(f"Unsupported normalization {normalization_name}")

            modules = [nn.Linear(in_features, units)]
            if normalization_cls is not None:
                modules.append(normalization_cls(units))
            modules.append(activation_cls())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

            self.hidden_layers.append(nn.Sequential(*modules))
            dims_in.append(in_features)
            dims_out.append(units)
            in_features = units

        self.output_layer = nn.Linear(in_features, n_classes)

        # === TAP REGISTRY ===
        self._tap_dims = {"pre0": n_features}
        L = len(dims_out)
        for i in range(1, L + 1):
            self._tap_dims[f"pre{i}"] = dims_in[i - 1]
            self._tap_dims[f"post{i}"] = dims_out[i - 1]
        self._tap_dims["pre_out"] = dims_out[-1]
        self._tap_dims["post_out"] = n_classes

        order_seq = ["pre0"]
        for i in range(1, L + 1):
            order_seq += [f"pre{i}", f"post{i}"]
        order_seq += ["pre_out", "post_out"]
        self._tap_order = {name: idx for idx, name in enumerate(order_seq)}

        # === RESIDUAL SETUP ===
        self._skips_to: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._skip_projs = nn.ModuleDict()
        for k, spec in enumerate(residual_specs or []):
            src_pos, src_idx = spec["from"]
            dst_pos, dst_idx = spec["to"]
            src_key, dst_key = f"{src_pos}{src_idx}", f"{dst_pos}{dst_idx}"

            if src_key not in self._tap_dims or dst_key not in self._tap_dims:
                raise ValueError(f"Unknown tap in residual {spec}")
            if self._tap_order[src_key] >= self._tap_order[dst_key]:
                raise ValueError(f"Invalid skip (backwards) {src_key}->{dst_key}")

            proj_name = f"proj_{k}"
            in_dim, out_dim = self._tap_dims[src_key], self._tap_dims[dst_key]
            self._skip_projs[proj_name] = (
                nn.Linear(in_dim, out_dim, bias=False)
                if in_dim != out_dim
                else nn.Identity()
            )
            self._skips_to[dst_key].append((proj_name, src_key))

    # === FORWARD ===
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        taps: dict[str, torch.Tensor] = {"pre0": x}
        h = x

        for i, block in enumerate(self.hidden_layers, start=1):
            pre_key, post_key = f"pre{i}", f"post{i}"

            base = h
            for proj_name, src_key in self._skips_to.get(pre_key, []):
                base = base + self._skip_projs[proj_name](taps[src_key])
            taps[pre_key] = base

            h = block(base)
            for proj_name, src_key in self._skips_to.get(post_key, []):
                h = h + self._skip_projs[proj_name](taps[src_key])
            taps[post_key] = h

        for proj_name, src_key in self._skips_to.get("pre_out", []):
            h = h + self._skip_projs[proj_name](taps[src_key])

        logits = self.output_layer(h)
        taps["post_out"] = logits

        for proj_name, src_key in self._skips_to.get("post_out", []):
            logits = logits + self._skip_projs[proj_name](taps[src_key])
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(5, 10)

    print("=== Demo 1: simple FNN ===")
    model1 = FeedforwardNetwork(
        n_classes=3,
        n_features=10,
        hidden_layers=[32, 16, 8],
    )
    y1 = model1(x)
    print("Output shape:", y1.shape)

    print("\n=== Demo 2: with residuals ===")
    residuals = [
        {"from": ("pre", 1), "to": ("post", 1)},  # within layer 1
        {"from": ("pre", 0), "to": ("post", 3)},  # input -> after layer 3
    ]
    model2 = FeedforwardNetwork(
        n_classes=3,
        n_features=10,
        hidden_layers=[32, 16, 8],
        residual_specs=residuals,
    )
    y2 = model2(x)
    print("Output shape:", y2.shape)
    print("Defined residuals:")
    for dst, lst in model2._skips_to.items():
        for proj_name, src in lst:
            print(f"  {src:>7s} → {dst:7s}  (proj: {proj_name})")

    print(model2)
