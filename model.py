import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('output/reps_clean.csv')
features = ["thigh", "knee", "hip", "lean", "knee_fwd", "down", "up"]

df = df.dropna(subset=features)

X = df[features]
y = df["quality"] == "bad"
groups = df["subject"]


cv = GroupKFold(n_splits=5)
model = make_pipeline(StandardScaler(), LogisticRegression())

scores = cross_val_score(model, X, y, cv=cv, groups=groups)

model.fit(X, y)
weights = model.named_steps["logisticregression"].coef_[0]
print(pd.Series(weights, index=features).sort_values())

print(X.shape)
print(y.mean())
print(groups.nunique())
print(scores)
print(scores.mean())


