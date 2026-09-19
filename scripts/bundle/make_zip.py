"""Zip a bundle directory so macOS and Linux keep the launcher executable.

PowerShell's ``Compress-Archive`` records no Unix permissions, so a ``run.command`` unzipped by
the macOS Archive Utility is not executable and Finder refuses to open it. This writes every
entry with Unix mode bits (0755 for launchers, 0644 otherwise); Windows ignores them.

    python scripts/bundle/make_zip.py <bundle_dir> <out.zip>
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

EXECUTABLE = {"run.command", "run.sh"}


def make_zip(bundle_dir: Path, out: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            arc = f"{bundle_dir.name}/{path.relative_to(bundle_dir).as_posix()}"
            info = zipfile.ZipInfo.from_file(path, arc)
            mode = 0o755 if path.name in EXECUTABLE else 0o644
            info.external_attr = (0o100000 | mode) << 16
            info.create_system = 3  # Unix, so unzip tools honour the mode bits
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as src, zf.open(info, "w") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)


if __name__ == "__main__":
    make_zip(Path(sys.argv[1]), Path(sys.argv[2]))
