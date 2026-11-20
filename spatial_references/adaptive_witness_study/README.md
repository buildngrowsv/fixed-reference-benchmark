# A 3D write gate changed confirmation direction when faults rotated

On a **262,144-voxel grid**, the controller recovered **70.38%** of stored 32-bit words after a fault rotation by routing to a different witness. Keeping the direction chosen during calibration recovered **52.67%**. Every policy accepted **99.00%** of legitimate writes. The adaptive choice used recent co-activation of the primary detector and no fault labels.

![Post-fault word recovery with fixed, calibrated and adaptive witness directions](adaptive-witness.png)

## How the controller chooses

Each write requires two positive ticks from a trained primary detector and a witness on the second tick. On a periodic 64³ grid the witness can be eight cells away along any coordinate axis. An adaptive controller receives the frozen primary detector's binary activity field. For direction `d`, it computes `C(d) = mean(a(x) × a(x+d))`. It updates `R(d) ← 0.92 R(d) + 0.08 C(d)` after each tick and takes the minimum direction on the next tick. It can use the first pulse when choosing the witness for the second pulse. The controller reads a full-grid summary and consults one remote witness at the write. It receives no ground-truth fault masks, stored words, or future observations. [Registered protocol](../ADAPTIVE_WITNESS.md) · [Implementation](../adaptive_witness.py)

Twelve independently generated worlds each contain a stable fault orientation and a matched variant that rotates the orientation halfway through 256 transitions. Faults occupy exactly a quarter of the voxels in both cases and imitate two-tick legitimate writes. Worlds share movement and intended writes. One percent of intended witness signals are absent. Three frozen versions of the original local detector and transport weights see every world. The comparator policies choose a balanced fixed axis, pick an axis once from the first four rounds of detector activity, or use a fixed perpendicular axis granted the hidden column orientation at every tick. The last comparator knows orientation and applies a deterministic perpendicular choice. It does not directly observe which specific columns are active. [Original learned gate](../README.md) · [Earlier fixed-witness result](../witness_study/README.md)

| Witness policy | Stable faults | Rotated faults | False writes after rotation |
|---|---:|---:|---:|
| Balanced fixed axis | 55.92% | 52.38% | 12.69% |
| Calibrate once | 65.74% | 52.67% | 15.70% |
| Choose from recent activity | 70.97% | 70.38% | 6.01% |
| Fixed perpendicular given the true orientation | 68.53% | 68.75% | 6.34% |

On rotated faults the adaptive policy gained **17.71 percentage points** over the calibration-only policy, with a paired 95% interval of **8.97 to 26.55 points**. It gained 18.00 points over a balanced fixed axis, interval 10.60 to 24.84. The recorded adaptive choices avoided the fault axis at every false-write decision in the measured rounds. It also chose between the two usable perpendicular directions using actual co-activation. This is a controlled synthetic result with axis-aligned, punctuated faults and a supplied candidate set. The adaptive policy receives a shared summary of the detector activity across the full grid. The comparison does not establish whether a local neighborhood can supply enough evidence when different regions have conflicting fault directions. [Score archives](results) · [World-level audit and intervals](audit.json)

Verify the checkpoint hashes, routing orientations, counts and paired bootstrap using `uv run --with numpy python spatial_references/adaptive_witness_study/verify.py`. The 74.97-second run reached 615 MB peak MLX allocation. It used an Apple M5 Ultra with 80 GPU cores, 256 GiB memory, macOS 27.2, Python 3.12.16, MLX 0.32.4 and NumPy 2.5.5. The controller evaluates states in float32. The owner proposed pursuing larger spatially structured references. OpenAI Codex designed, ran and verified the comparison and assisted with writing. [Execution receipt](results/receipt.json)

[Published Pubroot article](https://pubroot.com/ai/fine-tuning/a-3d-write-gate-changed-confirmation-direction-when-faults-rotated/) · [Submission and review](https://github.com/buildngrowsv/pubroot-website/issues/488)
