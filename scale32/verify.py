"""Check both sets of reported accuracies from saved predictions with NumPy."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent/"results"
r=json.loads((root/"summary.json").read_text())
b=json.loads((root/"bottleneck/summary.json").read_text())
for arm in ("fixed4","fixed32","writable32"):
    for split in ("natural","test"):
        with np.load(root/f"{split}-{arm}.npz") as z:
            a=(z["predictions"][:,:,1]==z["final"]).mean()
            o=(z["predictions"][:,:,0]==z["original"]).mean()
            np.testing.assert_allclose([a,o],[r["results"][arm][split][k] for k in ("accuracy","original")])
            if split=="test":assert (z["original"]==z["final"]).mean()==.5
    with np.load(root/"bottleneck"/(arm+".npz")) as z:
        np.testing.assert_allclose((z["predictions"]==z["truth"]).mean(axis=(0,1)),
                                   [b["results"][arm][k] for k in ("old","current")])
print("Verified the two endpoints for each arm, all three seeds, and the copy shortcut.")
