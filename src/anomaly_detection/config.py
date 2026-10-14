"""Load experiment configuration independently of the current working directory."""
from pathlib import Path
import yaml

def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    cfg["_config_file"] = str(path)
    return cfg

def resolve_path(cfg, value):
    """Resolve dataset/output paths against repository root, not CWD."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate
