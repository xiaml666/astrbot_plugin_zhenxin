#!/usr/bin/env python
"""Build the uploadable AstrBot plugin ZIP.

The archive contains one top-level folder:

    astrbot_plugin_zhenxin/
      main.py
      metadata.yaml
      _conf_schema.json
      logo.png
      requirements.txt
      src/zhenxin/...

Usage:
    python scripts/package_plugin.py
    # -> dist/astrbot_plugin_zhenxin-0.1.0.zip
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "astrbot_plugin_zhenxin"


def _version() -> str:
    text = (ROOT / "src" / "zhenxin" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("cannot read __version__ from src/zhenxin/__init__.py")
    return match.group(1)


VERSION = _version()


def _ignore(_dir, names):
    return {
        name
        for name in names
        if name in {"__pycache__", ".git", ".venv", "venv", "data", "logs", "reports", "tests", "dist", "models"}
        or name.endswith((".pyc", ".pyo", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm"))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "dist"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{PLUGIN_NAME}-{VERSION}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / PLUGIN_NAME
        stage.mkdir()
        for name in ("main.py", "metadata.yaml", "_conf_schema.json", "logo.png", "requirements.txt", "requirements-onnx.txt", "README.md", "LICENSE", "CHANGELOG.md"):
            src = ROOT / name
            if src.exists():
                shutil.copy2(src, stage / name)
        shutil.copytree(ROOT / "src" / "zhenxin", stage / "src" / "zhenxin", ignore=_ignore)
        for folder in ("pages", ".astrbot-plugin"):
            if (ROOT / folder).exists():
                shutil.copytree(ROOT / folder, stage / folder, ignore=_ignore)
        if (ROOT / "config.example.yaml").exists():
            shutil.copy2(ROOT / "config.example.yaml", stage / "config.example.yaml")
        scripts = ("astrbot_preflight.py", "validate_metrics.py", "lifetime_cost.py")
        (stage / "scripts").mkdir(exist_ok=True)
        for name in scripts:
            src = ROOT / "scripts" / name
            if src.exists():
                shutil.copy2(src, stage / "scripts" / name)
        (stage / "docs").mkdir(exist_ok=True)
        for name in ("validation.md", "astrbot-install.md", "astrbot-live-report.md", "cache-layout-rollout.md"):
            src = ROOT / "docs" / name
            if src.exists():
                shutil.copy2(src, stage / "docs" / name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage.parent).as_posix())

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    required = {
        f"{PLUGIN_NAME}/main.py",
        f"{PLUGIN_NAME}/metadata.yaml",
        f"{PLUGIN_NAME}/_conf_schema.json",
        f"{PLUGIN_NAME}/logo.png",
        f"{PLUGIN_NAME}/pages/zhenxin/index.html",
        f"{PLUGIN_NAME}/.astrbot-plugin/i18n/zh-CN.json",
        f"{PLUGIN_NAME}/src/zhenxin/engine.py",
    }
    missing = required - names
    if missing:
        raise SystemExit(f"zip missing: {sorted(missing)}")
    print(f"wrote {zip_path} ({zip_path.stat().st_size:,} bytes, {len(names)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
