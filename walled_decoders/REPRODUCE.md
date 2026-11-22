# Reproduce the walled decoder comparison

From the repository root, run `python3 walled_decoders/verify.py`. Its inputs are the saved Studio results and receipts. The check requires only the Python standard library.

To repeat training, use Apple Silicon with MLX. The pinned source is revision `4982b3395c2cace1c4295be8e0a47d93b099218a` from the research program. The corresponding files are in [source](source) and [source.zip](source.zip). Use Python 3.12 with `mlx==0.32.4`, `mlx-lm==0.32.2`, `numpy==2.5.5`, `transformers==5.23.0`, `tokenizers==0.23.3`, `pyarrow==26.0.1` and `huggingface-hub==2.4.0`.

Place the English, Welsh, Icelandic and Python corpus files in their dataset subfolders under `PROGRAM_DATA/raw`. The dataset revisions and file names are given in [the pinned download manifest](source/common/fetch.py), and the destinations and split rules are in [the data loader](source/common/textdata.py). For E3 use the FineWeb-Edu, FineWeb-2, CodeParrot Clean and CodeParrot Clean Valid entries. The exact benchmark tokenizer is [tokenizer.json](tokenizer.json), SHA-256 `01517103ad0ddc537ba180afb39e71564c072a8834e3982e4ce5d6816c2abf6b`. The preparation command trains the tokenizer and writes the shards. Its receipt records the digest so the regenerated file can be compared against the published tokenizer.

From `walled_decoders/source`, with the dependency environment active, use separate output directories for each run. The surgery step reads the main-run checkpoints from the named directories, so preserve them until it finishes.

```sh
export PROGRAM_DATA="$PWD/repro-data"
export STUDIO_QUEUE_ROOT="$PWD/repro-queue"
run_e3() {
  run_id="$1"
  shift
  STUDIO_JOB_DIR="$STUDIO_QUEUE_ROOT/runs/$run_id" python run.py "$@"
}
run_e3 prog-e3-data e3.prep
run_e3 prog-e3-screen e3.screen
run_e3 prog-e3-dense e3.arm --arm dense --seed 1
run_e3 prog-e3-walled-s1 e3.arm --arm walled --seed 1
run_e3 prog-e3-walled-s2 e3.arm --arm walled --seed 2
run_e3 prog-e3-topographic e3.arm --arm topographic --seed 1 --lam 0.05
run_e3 prog-e3-routed e3.arm --arm routed --seed 1
run_e3 prog-e3-surgery e3.surgery --refit-steps 400 --growth-steps 500
```

The supplied [evidence](evidence.json) includes reported wall-clock time, peak MLX allocation, training loss, per-domain ablation results, the refit control and growth outcomes. The FLOP estimate counts active block parameters, attention products and the output projection. Its formula is in [configs.py](source/e3/configs.py).
