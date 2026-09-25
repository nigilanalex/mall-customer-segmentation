"""Create a portable source/demo ZIP, excluding local environments and uploads."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

base = Path(__file__).resolve().parents[1]
destination = base.parent / f"{base.name}.zip"
excluded = {".git", ".venv", ".temp", ".qa", ".build", "__pycache__", "browser_checks"}
with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base)
        if not path.is_file() or any(part in excluded for part in relative.parts):
            continue
        if path.suffix in {".log", ".pyc", ".db"} or path.name.startswith((".env", "secrets.toml")) or ".db-" in path.name:
            continue
        archive.write(path, Path(base.name) / relative)
with ZipFile(destination) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Archive integrity check failed")
print(f"Created {destination} ({destination.stat().st_size:,} bytes)")
