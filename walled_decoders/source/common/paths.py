"""Where models, data and earlier job outputs live on the Studio."""
import os
from pathlib import Path

HOME = Path.home()
DATA = Path(os.environ.get("PROGRAM_DATA", HOME / "research" / "program-data"))
RAW = DATA / "raw"
TOKENS = DATA / "tokens"
QUEUE = Path(os.environ.get("STUDIO_QUEUE_ROOT", HOME / "research" / "architecture-queue"))
_CACHE = HOME / "research" / "routed-expert-splice" / "cache"
_LARGE = _CACHE / "large-opener"

# One shared copy of every pinned checkpoint. Revisions are Hugging Face commit hashes.
MODELS = {
    "qwen0.5b": _CACHE / "general",      # Qwen/Qwen2.5-0.5B base
    "qwen1.5b": _CACHE / "host",         # Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323
    "coder0.5b": _CACHE / "donor",       # Qwen/Qwen2.5-Coder-0.5B @ 8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301
    "qwen7b": _LARGE / "d149729398750b98c0af14eb82c78cfe92750796",
    "coder7b": _LARGE / "0396a76181e127dfc13e5c5ec48a8cee09938b02",
    "qwen14b": _LARGE / "97e1e76335b7017d8f67c08a19d103c0504298c9",
    "qwen32b": _LARGE / "1818d35814b8319459f4bd55ed1ac8709630f003",
    "qwen72b": _LARGE / "efba10c8e54e91e0d9570ab5f7b51a958474d4cb",
    "olmoe": _LARGE / "6d84c48581ece794365f2b8e9cfb043c68ade9c5",  # allenai/OLMoE-1B-7B-0924
}


def model_path(name: str) -> Path:
    path = MODELS[name]
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"{name} is not downloaded at {path}")
    return path


def job_output(job_id: str) -> Path:
    """Directory of an earlier queued job, for jobs that build on its outputs."""
    return QUEUE / "runs" / job_id
