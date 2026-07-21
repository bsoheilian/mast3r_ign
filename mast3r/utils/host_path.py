"""
Translate container paths back to their host equivalents.

Volume mappings are read directly from the docker-compose file that is
already mounted inside the container, so there is a single source of truth.
"""

from __future__ import annotations
import os
from pathlib import Path


def _load_volume_map() -> list[tuple[str, str]]:
    """Parse volume mounts from the compose file and return [(container, host), ...]."""
    host_docker_dir = os.environ.get("HOST_DOCKER_DIR")
    host_root = os.environ.get("HOST_MAST3R_IGN_ROOT")
    if not host_docker_dir or not host_root:
        return []

    compose_file = Path("/mast3r_ign/docker/docker-compose-cuda.yml")
    if not compose_file.exists():
        return []

    import yaml  # PyYAML is available in the image via dust3r deps
    with compose_file.open() as f:
        cfg = yaml.safe_load(f)

    volumes = []
    try:
        raw = cfg["services"]["mast3r-cuda"]["volumes"]
    except KeyError:
        return []

    for entry in raw:
        parts = str(entry).split(":")
        if len(parts) < 2:
            continue
        host_rel, container = parts[0], parts[1].rstrip("/")
        # Resolve relative paths the same way docker compose does:
        # relative to the directory that contains the compose file.
        host_abs = str(Path(host_docker_dir, host_rel).resolve())
        volumes.append((container, host_abs))

    # Longer container prefixes must be checked first (most specific wins).
    volumes.sort(key=lambda x: len(x[0]), reverse=True)
    return volumes


_VOLUME_MAP: list[tuple[str, str]] | None = None


def container_to_host(container_path: str) -> str:
    """Return the host path corresponding to *container_path*.

    Handles both absolute and relative paths. Relative paths are resolved
    relative to the current working directory.

    Falls back to returning *container_path* unchanged if no mapping matches
    or the env vars are not set (e.g. running outside Docker).
    """
    global _VOLUME_MAP
    if _VOLUME_MAP is None:
        _VOLUME_MAP = _load_volume_map()

    # Resolve relative paths to absolute
    p = Path(container_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    p_str = str(p.resolve()).rstrip("/")
    
    for container_prefix, host_prefix in _VOLUME_MAP:
        if p_str == container_prefix or p_str.startswith(container_prefix + "/"):
            return host_prefix + p_str[len(container_prefix):]

    return container_path
