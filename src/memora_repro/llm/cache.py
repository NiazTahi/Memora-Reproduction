from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, namespace: str, payload: dict[str, Any]) -> Path:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / namespace / f"{digest}.json"

    def get(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        path = self.key_for(namespace, payload)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, namespace: str, payload: dict[str, Any], value: Any) -> None:
        path = self.key_for(namespace, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

