"""
utils/config.py
---------------
YAML-based configuration loader that supports dot-notation access and
optional override merging (child config inheriting from ``_base_``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from adversarial_robustness.utils.logger import get_logger

logger = get_logger(__name__)


class Config:
    """
    Lightweight dot-access wrapper around a nested dictionary loaded from YAML.

    Example
    -------
    >>> cfg = load_config("configs/default.yaml")
    >>> cfg.attacks.fgsm.epsilon
    0.03
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(v) if isinstance(v, dict) else v for v in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to a plain dict."""
        result: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if isinstance(val, Config):
                result[key] = val.to_dict()
            elif isinstance(val, list):
                result[key] = [v.to_dict() if isinstance(v, Config) else v for v in val]
            else:
                result[key] = val
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str, base_dir: Optional[str] = None) -> Config:
    """
    Load a YAML config file.

    If the file contains a ``_base_`` key, the referenced base config is
    loaded first and the current file's values are deep-merged on top.

    Parameters
    ----------
    path:     Path to the YAML config file.
    base_dir: Directory to search for ``_base_`` relative paths.
              Defaults to the directory containing *path*.

    Returns
    -------
    Config object with dot-notation access.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    search_dir = Path(base_dir) if base_dir else config_path.parent

    with open(config_path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Handle inheritance
    base_name = raw.pop("_base_", None)
    if base_name:
        base_path = search_dir / base_name
        logger.debug("Loading base config: %s", base_path)
        base_cfg = load_config(str(base_path), base_dir=str(search_dir))
        raw = _deep_merge(base_cfg.to_dict(), raw)

    logger.debug("Loaded config from: %s", config_path)
    return Config(raw)
