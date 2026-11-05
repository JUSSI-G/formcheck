import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
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
lr = make_pipeline(StandardScaler(), LogisticRegression())
rf = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=20, random_state=0)

models = {
    "logistic regression": lr,
    "random forest": rf,
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, groups=groups)
    print(name, scores.mean(), scores)

df["pred"] = cross_val_predict(lr, X, y, cv=cv, groups=groups)
df["correct"] = df["pred"] == y

per_subject = df.groupby("subject")["correct"].agg(["mean", "count"]).sort_values("mean")
print(per_subject)

print(df[df["subject"] == 18].groupby("quality")[features].median().round(2))