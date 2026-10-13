"""Repository-relative defaults independent of the caller's working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "manta.yaml"
DEFAULT_HARBOR_CONFIG = PROJECT_ROOT / "config" / "harbor.yaml"


def resolve_config_path(path: str | Path) -> Path:
    """Resolve repository-relative config filenames independent of cwd."""
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    repository_path = PROJECT_ROOT / config_path
    if repository_path.exists():
        return repository_path
    configuration_path = PROJECT_ROOT / "config" / config_path
    return configuration_path if configuration_path.exists() else repository_path
