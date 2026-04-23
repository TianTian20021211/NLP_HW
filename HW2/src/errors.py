from __future__ import annotations

import sys
from typing import NoReturn


def error(message: str) -> NoReturn:
    sys.stderr.write(f"{message}\n")
    raise SystemExit(1)
