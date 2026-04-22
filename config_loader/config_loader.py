from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml"):
    """Load YAML config using a path relative to project root by default."""
    if Path(config_path).is_absolute():
        resolved_path = Path(config_path)
    else:
        # Resolve relative paths from the repository root, not notebook cwd.
        project_root = Path(__file__).resolve().parents[1]
        resolved_path = project_root / config_path

    with resolved_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config