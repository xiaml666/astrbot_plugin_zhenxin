"""Single source of truth for the plugin version.

AstrBot shows ``metadata.yaml`` in the plugin list, while the plugin page used
to read a hard-coded constant from ``src/zhenxin/__init__.py``.  When only one
of them was updated (a real deployment bug), the page kept showing an old
version.  Now every consumer — the ``@register`` decorator, ``/page/status``,
``/zhenxin`` and packaging — asks this module, and this module reads the same
``metadata.yaml`` AstrBot reads.
"""

from __future__ import annotations

from pathlib import Path

# Used only when the package is installed without its AstrBot metadata file
# (e.g. ``pip install -e .`` for standalone use).  scripts/package_plugin.py
# checks that this stays equal to metadata.yaml.
FALLBACK_VERSION = "0.6.3"


def _find_metadata() -> Path | None:
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:5]:
        candidate = parent / "metadata.yaml"
        if candidate.is_file():
            return candidate
    return None


def _parse_version(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            return value
    return ""


_METADATA = _find_metadata()
_VERSION = _parse_version(_METADATA) if _METADATA else ""
VERSION_SOURCE = "metadata.yaml" if _VERSION else "fallback"

__version__ = _VERSION or FALLBACK_VERSION


def build_info() -> dict[str, str | bool]:
    """Small payload for the page's version consistency row."""
    return {
        "version": __version__,
        "source": VERSION_SOURCE,
        "fallback": FALLBACK_VERSION,
        "metadata_path": str(_METADATA) if _METADATA else "",
        "mismatch": bool(_VERSION) and FALLBACK_VERSION != _VERSION,
    }

