from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Union


def load_config(path: Union[str, Path]) -> ModuleType:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG_PATH = str(path)
    return module


def config_to_dict(cfg: ModuleType) -> Dict[str, Any]:
    values = {}
    for key in dir(cfg):
        if key.isupper():
            value = getattr(cfg, key)
            if isinstance(value, tuple):
                value = list(value)
            values[key] = value
    return values


def resolve_config_path(cfg: ModuleType, value: Union[str, Path]) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return str(cwd_path)

    config_dir = Path(getattr(cfg, "CONFIG_PATH", ".")).resolve().parent
    return str((config_dir / path).resolve())
