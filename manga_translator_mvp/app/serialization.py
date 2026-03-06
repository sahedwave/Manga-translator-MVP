from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return to_builtin(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value
