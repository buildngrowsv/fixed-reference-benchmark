"""Entry point for every queued job in the research program.

    python run.py e1.donor --seed 1
    python run.py e2.train --arm lane --tokens 100000000

The Studio worker starts jobs with its own interpreter. This script re-executes itself in
the program's environment (MLX, mlx-lm and pyarrow at pinned versions), then runs the named
module as a script with the remaining arguments.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = Path.home() / "research" / "research-program" / ".venv"

if (ENV / "bin" / "python").exists() and Path(sys.prefix).resolve() != ENV.resolve():
    os.execv(str(ENV / "bin" / "python"), [str(ENV / "bin" / "python"), str(Path(__file__).resolve()), *sys.argv[1:]])

if len(sys.argv) < 2:
    sys.exit(__doc__)
sys.path.insert(0, str(ROOT))
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
