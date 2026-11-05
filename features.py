from pathlib import Path
import numpy as np
import pandas as pd
from reps import rep_stats


def find_duplicate_subjects(df):
    good = df[df["quality"] == "good"]
    bad = df[df["quality"] == "bad"]

    pairs = good.merge(bad, on=["subject", "rep"], suffixes=("_good", "_bad"))
    pairs["same"] = np.isclose(pairs["thigh_good"], pairs["thigh_bad"])

    share = pairs.groupby("subject")["same"].mean()
    return share[share > 0.5].index.tolist()

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
dup_videos = fingerprints[mask].index.tolist()
print("duplicated videos:", dup_videos)

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

MAX_MISSING = 0.10
duplicates = find_duplicate_subjects(df)
print("duplicate subjects:", duplicates)

clean = df[(df["missing"] <= MAX_MISSING) & (~df["subject"].isin(duplicates))]
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