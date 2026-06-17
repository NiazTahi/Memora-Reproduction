from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def project_path(root: str | Path, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return Path(root) / path

