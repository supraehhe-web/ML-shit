"""Run several repository scripts one after another inside a single container.

Used so a Modal GPU container is started once for a sequence of related jobs.
Each entry is "<script> <args...>". The chain stops at the first failure.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    cmds = [c for c in sys.argv[1:] if c.strip()]
    if not cmds:
        print("usage: run_chain.py '<script> <args>' ...")
        return 2
    for c in cmds:
        parts = shlex.split(c)
        full = [sys.executable, str(HERE / parts[0]), *parts[1:]]
        print("CHAIN", " ".join(parts), flush=True)
        t0 = time.time()
        code = subprocess.call(full)
        print(f"CHAIN-EXIT {code} in {time.time() - t0:.1f}s", flush=True)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
