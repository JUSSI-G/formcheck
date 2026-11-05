import json
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

features = ["thigh", "knee", "hip", "lean", "knee_fwd", "down", "up", "heel_lift"]

df = pd.read_csv("output/reps_clean.csv").dropna(subset=features)
errors = pd.read_csv("labels/errors.csv")
df = df.merge(errors, on="subject", how="left")

is_good = df["quality"] == "good"
is_depth_bad = (df["quality"] == "bad") & (df["error"] == "depth")
depth_df = df[is_good | is_depth_bad]

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(depth_df[features], depth_df["quality"] == "bad")

scaler = model.named_steps["standardscaler"]
lr = model.named_steps["logisticregression"]

export = {
    "features": features,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "weights": lr.coef_[0].tolist(),
    "intercept": float(lr.intercept_[0]),
}

os.makedirs("app/public", exist_ok=True)
with open("app/public/depth_model.json", "w") as f:
    json.dump(export, f, indent=2)

print(json.dumps(export, indent=2))