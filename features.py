from pathlib import Path
import numpy as np
import pandas as pd
from reps import rep_stats

rows = []
for path in sorted(Path("output").glob("*.npz")):
    data = np.load(path)
    reps = rep_stats(data["landmarks"], data["fps"], data["width"], data["height"])

    parts = path.stem.split("_")

    for i, rep in enumerate(reps):
        per_rep = {"subject": int(parts[1]), "quality": parts[-2], "view": parts[-1], "rep": i + 1, **rep}
        rows.append(per_rep)

df = pd.DataFrame(rows)
df.to_csv("output/reps.csv", index=False)
print(df.groupby("quality")["thigh"].describe())

cols = ["subject", "quality", "rep", "thigh"]
print(df.sort_values("thigh")[cols].head(10))
print(df.sort_values("thigh")[cols].tail(10))