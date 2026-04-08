"""List and prepare datasets for the visualizer API (paths relative to process CWD)."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def resolve_under_cwd(path: Path | str, *, cwd: Path | None = None) -> Path:
    """Resolve path; must stay under cwd (default: Path.cwd())."""
    base = (cwd or Path.cwd()).resolve()
    p = Path(path)
    target = (base / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Path must be under project directory") from e
    return target


def resolve_datasets_root(root: str, *, cwd: Path | None = None) -> Path:
    base = (cwd or Path.cwd()).resolve()
    p = Path(root)
    out = (base / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        out.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="datasets root must be under project directory") from e
    if not out.is_dir():
        raise HTTPException(status_code=404, detail=f"Not a directory: {out}")
    return out


def list_csv_under_datasets(datasets_dir: Path) -> list[str]:
    """POSIX paths relative to ``datasets_dir``, sorted."""
    root = datasets_dir.resolve()
    out: list[str] = []
    for f in sorted(root.rglob("*.csv")):
        if not f.is_file():
            continue
        try:
            rel = f.resolve().relative_to(root)
        except ValueError:
            continue
        out.append(rel.as_posix())
    return out
