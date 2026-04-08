"""Filesystem sandbox utilities for the local chat server."""

from __future__ import annotations

from pathlib import Path


class FilesystemSandbox:
    """Constrain filesystem operations to a configurable root directory."""

    def __init__(self, root: Path) -> None:
        """Initialize the sandbox and ensure the root directory exists."""
        self.root = root.resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def set_root(self, root: Path) -> None:
        """Change sandbox root and ensure the new directory exists."""
        new_root = root.resolve(strict=False)
        new_root.mkdir(parents=True, exist_ok=True)
        self.root = new_root

    def _resolve_in_root(self, rel_path: str) -> Path:
        """Resolve a relative path and reject traversal outside the sandbox root."""
        candidate = (self.root / rel_path).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Path escapes configured root directory.") from exc
        return candidate

    def list_dir(self, rel_path: str = ".") -> list[dict[str, str]]:
        """List child entries for a directory path inside the sandbox."""
        target = self._resolve_in_root(rel_path)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {rel_path}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {rel_path}")

        rows: list[dict[str, str]] = []
        for p in sorted(target.iterdir(), key=lambda x: x.name.lower()):
            kind = "dir" if p.is_dir() else "file"
            rows.append({"name": p.name, "type": kind})
        return rows

    def read_file(self, rel_path: str, max_chars: int = 120_000) -> str:
        """Read UTF-8 file content from the sandbox, truncated by max_chars."""
        target = self._resolve_in_root(rel_path)
        if not target.exists():
            raise FileNotFoundError(f"File does not exist: {rel_path}")
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {rel_path}")
        text = target.read_text(encoding="utf-8")
        return text[:max_chars]

    def write_file(self, rel_path: str, content: str, create_dirs: bool = True) -> None:
        """Write UTF-8 text to a file inside the sandbox (overwrite mode)."""
        target = self._resolve_in_root(rel_path)
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def append_file(self, rel_path: str, content: str, create_dirs: bool = True) -> None:
        """Append UTF-8 text to a file inside the sandbox."""
        target = self._resolve_in_root(rel_path)
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content)
