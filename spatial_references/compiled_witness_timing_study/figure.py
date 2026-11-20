"""Plot synchronized, warmed detector latencies on the Studio."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
values = json.loads((root / "results/result.json").read_text())["median_ms"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.6, 3.4), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
bars = ax.barh(["Full field → NumPy", "Two gathers → NumPy", "Compiled MLX"], values,
               color=["#e8bc79", "#b6acf0", "#84dfc2"])
ax.invert_yaxis()
ax.set(xlabel="Median synchronized latency (ms)", title="Same decisions on all 48 detector frames", xlim=(0, 8))
ax.grid(axis="x", alpha=.18)
ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5)
fig.savefig(root / "latency.png", dpi=170)
