# Can a learned sensor adapter transplant a doorway detector into protected memory?

A frozen neural gate recognized all 64 binary sensor patterns of its original doorway. Wired directly to those sensors, it also solved all 2,048 held-out paired memory tasks. When the sensors were permuted and its input adapter had to learn their new order, the transplant solved zero. A fresh controller trained on the same 128 examples solved all 2,048. The result identifies an interface-learning failure for this training budget.

## From a doorway to a write decision

The donor was a six-input, 16-hidden-unit neural network trained to let signals cross a spatial boundary. Its Boolean rule compares two endpoint readings and accepts a second condition when both latches are open. In the recipient, opening the gate saves the latest 32-bit word. Each paired test changes just one final sensor bit. One member must take the final word and the other must return a different earlier word. Both complete words must match. The independent test doubles the number of candidate writes from eight to sixteen. [Original doorway study](https://pubroot.com/ai/fine-tuning/can-a-learned-gate-recognize-a-movable-doorway-from-both-sides/) · [Preregistered design](PLAN.md)

All four fitted recipients saw the same permuted readings. The learned, analytic and corrupted frozen gates each received a fully trainable affine adapter with a sigmoid output. The fresh controller learned directly from all sensors. Each fit used 120 Adam updates on minibatches from 128 distinct episodes and one training seed. The 512-pair validation gate required an advantage over both controls before an independent training replication. [Training and checkpoint code](run.py)

| Gate and wiring | Both correct at 8 writes, out of 512 | Both correct at 16 writes, out of 2,048 |
|---|---:|---:|
| Frozen trained gate, fitted adapter | 0 | 0 |
| Fresh trainable gate | 512 | 2,048 |
| Corrupted table, fitted adapter | 512 | 2,047 |
| Analytic rule, fitted adapter | 512 | 2,048 |
| Frozen trained gate, known correct wiring | 512 | 2,048 |

The correct-wiring row is a diagnostic added after the four planned arms completed. With the failed donor's adapter held fixed, substituting the corrupted table rescued 664 pairs and lost zero. Disabling the donor rescued zero. The corrupted gate's separately optimized adapter performed almost perfectly. A complicated interpolated table plus a trainable adapter can therefore approximate this recipient rule even after the original table was scrambled. The fresh model already reached the ceiling, so the planned replication closed. A larger training budget, different interface parameterization, or different task could change the comparison. [Results and interventions](results/result.json)

An independent NumPy implementation loaded the public checkpoints, checked that every pretrained donor weight retained its original value, ran all 64 source inputs, reconstructed both sensor worlds, and reproduced every saved individual success indicator on validation and test. Its direct-wiring diagnostic solves every pair. Run `python spatial_graft/verify.py spatial_graft/results hidden_door/results/both_ends-101.safetensors` with NumPy and safetensors. [Audit](results/audit.json) · [Predictions and weights](results) · [Audit implementation](verify.py)

The account owner proposed testing transfers between protected regions. OpenAI Codex implemented the study, ran it over SSH on the owner's Apple M5 Ultra, audited the individual outcomes and assisted with this manuscript.

[Pubroot submission and review record](https://github.com/buildngrowsv/pubroot-website/issues/360)
