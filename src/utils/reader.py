from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)  # -> звичайний Python-словник

    return _cfg

