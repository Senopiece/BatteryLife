from __future__ import annotations

from typing import Any, Mapping
import numbers

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in lightweight contexts
    torch = None

import trackio


def _sanitize_value(value: Any) -> Any:
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, numbers.Number) or isinstance(value, (str, bool)):
        return value
    return value


def init(**kwargs: Any):
    return trackio.init(**kwargs)


def log(metrics: Mapping[str, Any], step: int | None = None) -> None:
    clean = {k: _sanitize_value(v) for k, v in metrics.items()}
    if step is None:
        trackio.log(clean)
    else:
        trackio.log(clean, step=int(step))


def finish() -> None:
    trackio.finish()
