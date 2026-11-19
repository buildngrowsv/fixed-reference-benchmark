# Fixed-reference benchmark on Apple M5 Ultra

November 19, 2026. All data generation, model training, inference, and statistical comparisons ran on the Mac Studio over SSH. Each model makes one forward pass. The four source activations may change across examples and across training updates. Their values in the fixed region stay identical throughout the six commands of a particular pass. Gradients flow to the source encoder.

## Comparison on the balanced test

| Network | Final color on restore-sensitive boards | Final color on ordinary boards | Starting color on ordinary boards | Learned parameters | Training seconds per run |
|---|---:|---:|---:|---:|---:|
| Ordinary residual | 64.27% | 76.76% | 99.73% | 159,872 | 9.8 |
| Copy at output only | 64.00% | 76.63% | 99.84% | 153,728 | 10.0 |
| Writable copy | 64.21% | 76.26% | 99.75% | 159,872 | 11.4 |
| Fixed reference | 63.82% | 76.57% | 99.77% | 156,800 | 10.7 |

The primary measure uses 10,000 held-out examples in which omitting at least one restore changes the correct answer. Exactly half end at their queried square's starting color. Each of the four answer colors is equally frequent within both halves. Copying the starting color scores **50%**. Uniform random answers average 25%. The ordinary test column uses 10,000 separately sampled cases with the natural frequency of commands and outcomes.

| Fixed reference compared with | Difference in percentage points | Approximate 95% paired interval | Seed pairs won |
|---|---:|---:|---:|
| Ordinary residual | -0.45 | -2.05 to +1.00 | 2/5 |
| Copy at output only | -0.19 | -1.45 to +1.06 | 2/5 |
| Writable copy | -0.40 | -1.77 to +1.19 | 1/5 |

These are paired comparisons of the fixed version against each other version on the primary score. The approximate intervals resample both the five training seeds and the same boards used by all models, stratified by answer color and changed/unchanged status. Only five seeds are represented, and the intervals do not adjust for multiple comparisons.

## Does the protected region affect answers?

For the fixed network, we changed one color in the starting board while keeping commands and the queried square identical. The simulator chose a color that must change the final answer. For model-board pairs on which the unmodified model answered both versions correctly, we transplanted only the four source values from the second board into the network processing the first board. The final prediction switched to the donor board's answer in **2,177 of 23,727 cases (9.18%)**. It stayed at the first board's answer in 83.59%. This intervention shows when changing only that pathway redirects the answer. It can create internal states absent in training, and a change in an individual answer does not establish a population-level accuracy advantage.

## Original protocol and the shortcut

| Network | Final color on restore-sensitive boards | Final color on ordinary boards | Starting color on ordinary boards | Learned parameters | Training seconds per run |
|---|---:|---:|---:|---:|---:|
| Ordinary residual | 65.79% | 54.59% | 99.42% | 159,872 | 2.4 |
| Copy at output only | 65.94% | 54.85% | 99.52% | 153,728 | 2.7 |
| Writable copy | 65.84% | 54.43% | 99.33% | 159,872 | 2.6 |
| Fixed reference | 66.53% | 54.84% | 99.42% | 156,800 | 2.8 |

On the original restore-sensitive set, repeating the starting color scored **86.60%**. It scored 65.42% on ordinary cases. The pilot reached its 16-epoch cap and validation accuracy declined over training. All four models underperformed simple copying on this original restore-sensitive set. Its ranking is weak evidence about the architecture.

| Fixed reference compared with | Difference in percentage points | Approximate 95% paired interval | Seed pairs won |
|---|---:|---:|---:|
| Ordinary residual | +0.73 | -0.18 to +1.72 | 4/5 |
| Copy at output only | +0.58 | -0.15 to +1.31 | 5/5 |
| Writable copy | +0.69 | -0.37 to +1.71 | 3/5 |

## What the follow-up changed

The follow-up used fresh board-and-command sequences, disjoint from every split in the original study. It still used 50,000 training boards and the same six commands, architectures, channel widths, and paired seeds. Each variant saw the same minibatches in the same order within a seed. In the original run, one sampled square per board supplied the starting and final labels. In the follow-up, all eight squares supplied both labels during training. We weighted final-color errors more heavily and doubled the current-color loss weight on squares whose final color differed from their initial color. The learning rate changed from 0.002 to 0.001. Only the final readout sees the query. A separate ordinary-model validation pilot chose a single budget of 24 epochs for all four follow-up versions. The two protocols have different data, supervision and budgets, so improvements across protocols cannot be attributed to any one change.

Each eight-block network uses a learned board encoder, six command-conditioned residual updates, and a readout. The 64-value fixed model has 60 mutable working values and four protected values. The endpoint model uses 60 working values and can consult the protected four at its readout. The writable version starts with the same predictions as the fixed model and learns whether to modify its four slots. The ordinary network can update all 64 slots from the start. These designs match the width and optimization budget and have the slightly different parameter counts reported above. Training timings include optimization and exclude validation and test scoring.

Generator and architecture checks verified disjoint splits, the exact restore counterfactual against an independent simulator, unchanged reference bits, equivalent fixed/writable predictions at initialization, and the endpoint model's lack of intermediate access.

## Reproduce or inspect

The local folder contains the exact scripts, generated datasets, per-seed learning curves, predictions, environment versions, and summaries. To rerun from scratch, copy the three Python scripts into a new directory on the Studio. Use the following commands from that directory.

```sh
for phase in prepare pilot run summarize; do
  ~/.local/bin/uv run --python 3.12 benchmark.py "$phase" || break
done
for phase in prepare pilot run summarize; do
  ~/.local/bin/uv run --python 3.12 balanced.py "$phase" || break
done
```

[Original per-seed scores](width-64/per-seed.csv) · [Balanced per-seed scores](balanced/width-64/per-seed.csv) · [Balanced summary and paired intervals](balanced/width-64/summary.json)
