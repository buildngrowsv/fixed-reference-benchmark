"""One-time downloads for the research program, at pinned dataset and model revisions."""
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
RAW = Path.home()/"research/program-data/raw"
FILES = [
  ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9", "data/cym_Latn/test/000_00000.parquet"),
  ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9", "data/cym_Latn/train/000_00000.parquet"),
  ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9", "data/isl_Latn/test/000_00000.parquet"),
  ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "sample/10BT/000_00000.parquet"),
  ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy/test-00000-of-00001.parquet"),
  ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data/validation-00000-of-00001.parquet"),
  ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9", "data/isl_Latn/train/000_00000.parquet"),
  ("codeparrot/codeparrot-clean", "35a59fb025bc0a102f7d96eac09d145b896d487b", "file-000000000001.json.gz"),
  ("codeparrot/codeparrot-clean", "35a59fb025bc0a102f7d96eac09d145b896d487b", "file-000000000002.json.gz"),
  ("codeparrot/codeparrot-clean-valid", "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd", "file-000000000054.json.gz"),
]
for repo, rev, name in FILES:
    p = hf_hub_download(repo, name, revision=rev, repo_type="dataset", local_dir=RAW/repo.split("/")[1])
    print("ok", p, flush=True)
OLMOE = ("allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5")
dest = Path.home()/"research/routed-expert-splice/cache/large-opener"/OLMOE[1]
snapshot_download(OLMOE[0], revision=OLMOE[1], local_dir=dest, allow_patterns=["*.json","*.safetensors"])
(dest/".downloaded").touch()
print("ok", dest, flush=True)
