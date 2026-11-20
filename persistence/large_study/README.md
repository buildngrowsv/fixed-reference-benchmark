# More local routing capacity produces a modest gain in neutral two-bank spatial memory

Increasing a local controller from 468 to 57,812 learned weights raised exact 32-bit word recall from 1,071 to 1,247 of 18,544 words on the same held-out three-dimensional volumes. A controller initialized with separate working and snapshot roles recalled all 18,544 words at both sizes. Extra capacity improved the neutral model by 0.95 percentage points, while its final 6.72% recall remained far below the 90% criterion for testing larger volumes.

Both controllers stored two 32-value banks in each cell. They could read the same 14 local ports, eight sensor bits and eight commands. The larger rule adds a trainable routing correction indexed by the command and sensor pattern. These corrections start at zero, giving matching initial transitions in the small and large controllers. Parameter count and the form of the local rule both change. All arms used the same seed-101 batches, 500 updates, 4 × 4 × 4 training grids and eight update steps per rollout. Evaluation used 6 × 6 × 6 grids over 16 steps. [Registered comparison](../LARGE_CONTROLLER.md) · [Earlier spatial memory study](../README.md)

| Initial bank roles | 468 weights | 57,812 weights | Additional correct words |
|---|---:|---:|---:|
| Neutral, persistent banks | 5.78% | 6.72% | 176 |
| Working and snapshot preferences | 100% | 100% | 0 |

The interval on the neutral improvement was +0.05 to +1.84 percentage points, bootstrapping pairs of test volumes. We also corrupted each bank separately by substituting its contents from another volume. Validation determined which bank should carry the memory. On test, corrupting that bank hurt the neutral controller only 1.06 points more than corrupting its partner. For the role-initialized controller the difference was 100 points. These measurements cover one training draw and one conditional parameterization. The prespecified 4,096-cell stress test did not run because the neutral controller missed its accuracy and selective-disruption thresholds. [Raw counts and receipt](../results/large-controller) · [Independent audit](verify.py)

The owner proposed revisiting spatial memory at larger network sizes. OpenAI Codex implemented the comparison, ran it on the owner's Mac Studio, checked the published counts and assisted with writing. [Reconstructed counts](audit.json)
