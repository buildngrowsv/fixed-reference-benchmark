# Domain selectivity and room transplantation in walled language models

We trained decoders with hard walls, a wiring-length penalty, learned routing and dense connectivity. Each main run used 400 million tokens. Both hard-wall seeds approached dense validation loss, with an estimated 23% higher compute per token. No main-model room, contiguous arc or expert met the registered domain-selectivity threshold. A transplanted room followed by door refitting recovered 67.0% of the code-loss damage, slightly above 64.3% for door refitting with the room still removed. The registered outcome was a null for the wall hypothesis.

## Training and removal

All models used RoPE and SwiGLU with an 8,192-entry byte-level BPE tokenizer fit to English prose, Python and Welsh. Each batch contained 40 sequences of 512 tokens from each domain. Five small runs used roughly 200M tokens per arm. The hard-wall and wiring-penalty arms missed the screen threshold. The registered plan continued to the larger comparison to test whether selectivity appeared with more parameters and training.

At the larger size, four rooms in the walled model could exchange information through 32 shared channels. Each room had its own block weights. The topographic comparator used a circular wiring-length penalty. The routed comparator kept dense attention and selected two of four feed-forward experts per token. All five main runs used 399,974,400 tokens. The four configurations at seed 1 shared the batch order. The second walled seed changed initialization and batch order. Reported loss averages English, Python and Welsh equally. [Registered protocol and source](source.zip)

| Main arm | Total parameters | Mean loss | Best selectivity | Estimated FLOPs/token |
|---|---:|---:|---:|---:|
| Dense | 97.5M | 2.5437 | 0.075 | 604.0M |
| Hard walls, seed 1 | 117.7M | 2.5504 | 0.280 | 742.4M |
| Hard walls, seed 2 | 117.7M | 2.5681 | 0.091 | 742.4M |
| Wiring penalty | 97.5M | 2.5505 | 0.047 | 604.0M |
| Learned routing | 154.2M | 2.5141 | 0.284 | 604.2M |

The removal tests silenced each walled room and each routed expert, and ablated four contiguous arcs in the other models. A room counted when its removal increased its most affected domain’s loss by at least 1%. A selectivity score of 0.5 required that increase to be at least three times the mean increase on the other two domains. All five main models fell below 0.5. The small routed model reached 0.673, an effect that did not persist in the larger run. The two walled seeds cost 0.26% and 0.96% in mean loss at equal tokens. Wider embeddings, shared doors and different estimated work per token limit comparisons at fixed compute.

## Room transplant and growth

We took the room in seed 1 whose removal hurt code the most relative to other domains. We replaced its block weights and its slices of the embedding, unembedding and final norm with the matching room from seed 2. Both the transplanted and removed-room models got the same 400 updates to door weights alone. Code loss was 1.852 in the intact model, 2.469 after removal and 2.581 after the transplant before refitting. After refitting it was 2.056 with the transplant and 2.072 with the room still removed. The 2.66-point recovery advantage fell below the registered 25-point margin and the transplant fell below the 90% absolute recovery target.

Adding a fifth room almost preserved the initial validation losses, with maximum drift of 0.0000025 nats per token on the three training domains. A separate 500-update comparison adapted to Icelandic, which the original model had never seen.

| Trainable part during Icelandic adaptation | Icelandic gain | English loss rise | Python loss rise | Welsh loss rise |
|---|---:|---:|---:|---:|
| New room only | 2.140 | 0.127 | 0.229 | 0.703 |
| New room and shared door weights | 2.294 | 0.315 | 0.365 | 1.420 |
| Full fine-tuning | 2.486 | 0.449 | 0.511 | 1.909 |

Training the new room alone recovered 86.1% of full fine-tuning’s Icelandic gain with smaller loss increases on each old domain. In that arm, the existing door weights stayed frozen and the new room’s trainable projections could still affect shared channels. It tests selective weight updates within the walled architecture. The three methods used their registered learning rates and training settings.

## Interpretation and evidence

Hard connectivity constraints did not make the resulting rooms replaceable domain specialists at this budget. The shared door refit explains most of the observed transplant recovery. The result concerns this tokenizer, mixture and one donor–recipient seed pair. TopoLM uses a spatial smoothness objective on activations. The circular wiring-length penalty tested here follows a different choice closer to brain-inspired modular training. [TopoLM](https://arxiv.org/abs/2410.11516) · [Brain-inspired modular training](https://arxiv.org/abs/2305.08746)

The Studio executed revision `4982b3395c2cace1c4295be8e0a47d93b099218a`. Run `python3 walled_decoders/verify.py` to recalculate the main losses, removal selectivity, transplant recoveries and growth changes from the saved reports. The source archive includes the registered protocol and runnable implementation at the execution revision. [Evidence and receipts](evidence.json) · [Verifier](verify.py) · [Source archive](source.zip)
