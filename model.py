import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('output/reps_clean.csv')
features = ["thigh", "knee", "hip", "lean", "knee_fwd", "down", "up"]

df = df.dropna(subset=features)

X = df[features]
y = df["quality"] == "bad"
groups = df["subject"]

print(X.shape)
print(y.mean())
print(groups.nunique())


