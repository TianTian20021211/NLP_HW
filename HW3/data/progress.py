"""Small optional progress helpers for long-running data loaders."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, TypeVar


T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback only used without tqdm installed
    _tqdm = None


class _PlainProgress:
    """Small ASCII progress bar used when tqdm is not installed."""

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        total: int | None = None,
        desc: str = "",
        unit: str = "it",
        unit_scale: bool = False,
        mininterval: float = 0.2,
        **_: object,
    ) -> None:
        self.iterable = iterable
        self.total = total if total is not None else self._safe_len(iterable)
        self.desc = desc
        self.unit = unit
        self.unit_scale = unit_scale
        self.mininterval = mininterval
        self.n = 0
        self._last_render = 0.0
        self._postfix = ""
        self._closed = False

    @staticmethod
    def _safe_len(iterable: Iterable[T] | None) -> int | None:
        if iterable is None:
            return None
        try:
            return len(iterable)  # type: ignore[arg-type]
        except TypeError:
            return None

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            return iter(())

        def _generator() -> Iterator[T]:
            try:
                for item in self.iterable:
                    yield item
                    self.update(1)
            finally:
                self.close()

        return _generator()

    def __enter__(self) -> "_PlainProgress":
        self._render(force=True)
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.n += n
        self._render()

    def set_postfix(self, *args: object, **kwargs: object) -> None:
        parts: list[str] = []
        if args and isinstance(args[0], dict):
            parts.extend(f"{k}={v}" for k, v in args[0].items())
        parts.extend(f"{k}={v}" for k, v in kwargs.items())
        self._postfix = " ".join(parts)
        self._render(force=True)

    def close(self) -> None:
        if self._closed:
            return
        self._render(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._closed = True

    def _render(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_render) < self.mininterval:
            return
        self._last_render = now

        prefix = f"{self.desc}: " if self.desc else ""
        if self.total:
            ratio = min(1.0, self.n / self.total)
            width = 28
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            line = (
                f"\r{prefix}|{bar}| {ratio:6.1%} "
                f"{self._format(self.n)}/{self._format(self.total)} {self.unit}"
            )
        else:
            line = f"\r{prefix}{self._format(self.n)} {self.unit}"

        if self._postfix:
            line += f" [{self._postfix}]"

        sys.stderr.write(line)
        sys.stderr.flush()

    def _format(self, value: int) -> str:
        if not self.unit_scale:
            return f"{value:,}"
        scaled = float(value)
        for suffix in ("", "K", "M", "G", "T"):
            if abs(scaled) < 1000 or suffix == "T":
                return f"{scaled:.1f}{suffix}"
            scaled /= 1000


def progress(iterable: Iterable[T] | None = None, **kwargs: object):
    """Return tqdm when available, otherwise use the built-in ASCII bar."""
    if _tqdm is None:
        return _PlainProgress(iterable, **kwargs)
    return _tqdm(iterable, **kwargs)


class _ProgressReader:
    """File wrapper that updates a byte progress bar as pandas reads."""

    def __init__(self, f: BinaryIO, bar: _PlainProgress) -> None:
        self._f = f
        self._bar = bar

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        self._bar.update(len(data))
        return data

    def readline(self, size: int = -1) -> bytes:
        data = self._f.readline(size)
        self._bar.update(len(data))
        return data

    def readinto(self, b: bytearray) -> int:
        n = self._f.readinto(b)
        self._bar.update(n)
        return n

    def readable(self) -> bool:
        return True

    def __iter__(self) -> Iterator[bytes]:
        for line in self._f:
            self._bar.update(len(line))
            yield line

    def __getattr__(self, name: str) -> object:
        if name == "fileno":
            raise AttributeError(name)
        return getattr(self._f, name)


@contextmanager
def progress_file_reader(path: Path, desc: str):
    """Open a file while tracking bytes consumed by direct read calls."""
    if _tqdm is None:
        with path.open("rb") as f, _PlainProgress(
            total=path.stat().st_size,
            desc=desc,
            unit="B",
            unit_scale=True,
        ) as bar:
            yield _ProgressReader(f, bar)
        return

    with path.open("rb") as f:
        with _tqdm.wrapattr(
            f,
            "read",
            total=path.stat().st_size,
            desc=desc,
            unit="B",
            unit_scale=True,
        ) as wrapped:
            yield wrapped
