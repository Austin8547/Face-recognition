import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)