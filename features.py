from pathlib import Path
import numpy as np
import pandas as pd
from reps import rep_stats


rows = []
for path in sorted(Path("output").glob("*.npz")):
    data = np.load(path)
    reps = rep_stats(data["landmarks"], data["fps"], data["width"], data["height"])

    parts = path.stem.split("_")
    missing = np.isnan(data["landmarks"][:, 23, 1]).mean()

    for i, rep in enumerate(reps):
        per_rep = {"subject": int(parts[1]), "quality": parts[-2], "view": parts[-1], "rep": i + 1, **rep, "missing": missing}
        rows.append(per_rep)

df = pd.DataFrame(rows)
fingerprints = df.groupby(["subject", "quality"])["thigh"].apply(lambda s: tuple(s.round(4)))
df.to_csv("output/reps.csv", index=False)

mask = fingerprints.duplicated(keep=False)

dups = fingerprints[mask]
drop = []

for fp, group in dups.groupby(dups):
    videos = group.index.tolist()
    qualities = {quality for subject, quality in videos}
    if len(qualities) > 1:
        drop += videos
    else:
        drop += videos[1:]

print("dropping:", drop)

video_ids = pd.MultiIndex.from_frame(df[["subject", "quality"]])

MAX_MISSING = 0.10

clean = df[(df["missing"] <= MAX_MISSING) & (~video_ids.isin(drop))]
print(len(df), "reps total,", len(clean), "after filtering")
print(clean.groupby("quality")["thigh"].describe())

cols = ["subject", "quality", "rep", "thigh"]
print(clean.sort_values("thigh")[cols].head(10))
print(clean.sort_values("thigh")[cols].tail(10))

for threshold in [-30, -25, -20, -15, -10]:
    predicted_bad = clean["thigh"] < threshold
    actual_bad = clean["quality"] == "bad"
    accuracy = (predicted_bad == actual_bad).mean()
    print(f"threshold {threshold}°: accuracy {accuracy:.0%}")

features = ["thigh", "knee", "hip", "lean", "knee_fwd", "down", "up"]
print(clean.groupby("quality")[features].median().round(2))

clean.to_csv("output/reps_clean.csv", index=False)