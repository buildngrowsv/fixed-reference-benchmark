# Fixed-reference benchmark

Does keeping four early activations read-only help a feed-forward network edit a board and recover its original state? A comparison across five paired seeds found **63.82%** final-color accuracy for a fixed reference and **64.21%** for an initially identical writable copy on 10,000 balanced restore-sensitive cases. The paired difference was **−0.40 percentage points**, with an approximate 95% interval from **−1.77 to +1.19**.

[Read the article](paper.md) · [Published original study](https://pubroot.com/data/statistics/testing-read-only-activations-in-a-feed-forward-board-editing-network/) · [Experiment report](Results.md) · [Per-seed results](balanced/width-64/per-seed.csv) · [Confidence intervals](balanced/width-64/summary.json)

## Larger-scale follow-up

[Width-256 experiment and forced-memory diagnostic](scale32/README.md) · [Study plan](scale32/PLAN.md) · [Published wider-reference study](https://pubroot.com/data/statistics/a-wider-reference-solved-delayed-recall-but-left-board-editing-below-a-copy-shortcut/)

## Spatial routing and intermediate snapshots

[Experiments on 32-bit spatial memory](spatial_references/README.md) compare signed and bounded writes, an open or closed doorway, and a learned gate that stores intermediate rather than initial values.

## Learning where to store a snapshot

[Follow-up on interchangeable spatial banks](learned_banks/README.md) compares assigned and learnable bank roles at equal state capacity and measures which stored values remain causally available.

## Check the results without training

From this repository, with Python package runner [uv](https://docs.astral.sh/uv/) installed, run the following command. It needs NumPy and works without Apple hardware.

```sh
uv run --python 3.12 verify_results.py
```

The check reads saved predictions and test labels for all 40 runs, recomputes accuracy and the paired intervals, and recounts the reference-swap intervention.

## Train from scratch

Training uses Apple MLX on an Apple Silicon Mac with a supported Metal GPU. Copy the three experiment files into a fresh directory so that the resume logic starts new runs. The recorded environment is in [runtime.json](runtime.json). From the repository root, run:

```sh
set -e
run_dir=$(mktemp -d)
cp benchmark.py balanced.py analyze.py "$run_dir/"
cd "$run_dir"
for phase in prepare pilot run summarize; do
  uv run --python 3.12 benchmark.py "$phase"
done
for phase in prepare pilot run summarize; do
  uv run --python 3.12 balanced.py "$phase"
done
```

The original protocol must run first because the follow-up excludes its board-and-command sequences. Both use an ordinary-network validation pilot to set the common training budget for the four variants.

## Evidence

The original splits are in `data/` with predictions and learning curves in `width-64/`. The independent follow-up uses the same layout under `balanced/`. Each seed/variant directory contains a result JSON and compressed predictions. The synthetic test labels, analysis code and training scripts are included. Five seeds, one task and one network width limit the scope of the result.

The original architectural question came from the account owner. OpenAI Codex implemented the experiments, ran them on the owner's Mac Studio, analyzed the predictions and assisted with the manuscript.

## Do the banks need distinct roles?

Giving both interchangeable banks the same preference for retention recovered 5.78% of exact words. The same architecture with a role-specific start recovered every held-out word. [Controlled pilot, weights and causal intervention](persistence/README.md) · [Research decisions](RESEARCH.md)
