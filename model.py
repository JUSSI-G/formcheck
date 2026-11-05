import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

features = ["thigh", "knee", "hip", "lean", "knee_fwd", "down", "up", "heel_lift"]


def evaluate(data, title):
    X = data[features]
    y = data["quality"] == "bad"
    groups = data["subject"]
    cv = GroupKFold(n_splits=5)

    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
        "random forest": RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=20, random_state=0),
    }

    print(f"\n{title}: {len(data)} reps, {y.mean():.0%} bad, {groups.nunique()} subjects")

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="balanced_accuracy")
        print(f"  {name}: {scores.mean():.0%}  {[f'{s:.2f}' for s in scores]}")

    pred = cross_val_predict(models["logistic regression"], X, y, cv=cv, groups=groups)
    correct = pd.Series(pred == y, index=data.index)
    print(correct.groupby(data["subject"]).agg(["mean", "count"]).sort_values("mean").head(6))


df = pd.read_csv("output/reps_clean.csv")
df = df.dropna(subset=features)

errors = pd.read_csv("labels/errors.csv")
df = df.merge(errors, on="subject", how="left")

is_good = df["quality"] == "good"
is_depth_bad = (df["quality"] == "bad") & (df["error"] == "depth")
depth_df = df[is_good | is_depth_bad]

evaluate(df, "all errors")
evaluate(depth_df, "depth only")