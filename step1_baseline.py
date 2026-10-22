"""
step1_baseline.py  --  Step 1: build a trustworthy measuring stick.

The story: before we ask "how small can the detector get?", we need a ruler that
does not lie. A single random split with one seed can flatter a model by luck, and
rows that are copies of each other can leak from training into testing. This script
(a) describes the data, (b) checks for leakage, and (c) scores four simple models
with repeated cross-validation, so every later experiment has a baseline to beat.

Run from C:\\Biomed_project inside the venv:   python step1_baseline.py
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

SEED = 42

# ---------------------------------------------------------------- 1. Load data
df = pd.read_csv("BEED_Data.csv")
X = df.drop(columns=["y"]).to_numpy()
y = (df["y"] > 0).astype(int).to_numpy()   # 0 = non-seizure, 1 = seizure (same as main.py)

print("=== 1. The data ===")
print("rows, columns          :", df.shape)
print("original class counts  :", df["y"].value_counts().sort_index().to_dict())
print("binary counts (0 / 1)  :", np.bincount(y).tolist())
print("always-seizure accuracy:", round(y.mean(), 4), "(the number any model must beat)")

# How many bits does a signed integer comparator need for these inputs?
# This sets the starting point for the later bit-width experiment.
biggest = int(np.abs(X).max())
bits_needed = math.ceil(math.log2(biggest + 1)) + 1      # +1 for the sign bit
print("input min / max        :", X.min(), "/", X.max())
print("signed bits needed     :", bits_needed, "(your Verilog uses 16)")

# ------------------------------------------------------- 2. Leakage / duplicates
print("\n=== 2. Duplicate check ===")
feature_cols = list(df.columns[:-1])
n_dup = int(df.duplicated(subset=feature_cols).sum())
print("rows whose 16 values exactly repeat another row:", n_dup)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

# For the first fold only: how many test rows have an exact twin in the training set?
train_idx, test_idx = next(cv.split(X, y))
train_rows = {row.tobytes() for row in X[train_idx]}
leaked = sum(row.tobytes() in train_rows for row in X[test_idx])
print(f"test rows with an exact twin in training (fold 1): {leaked} of {len(test_idx)}")
print("(near-duplicates would not show up here, so treat 0 as 'no exact copies', not 'no leakage')")

# ------------------------------------------------------ 3. Score the baselines
def evaluate(name, make_model):
    """Train and test a fresh model on every fold; return mean and spread of each metric."""
    recalls, specs, accs = [], [], []
    for tr, te in cv.split(X, y):
        model = make_model()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        recalls.append(tp / (tp + fn))     # seizure recall: seizures caught
        specs.append(tn / (tn + fp))       # specificity: non-seizures correctly left alone
        accs.append((tp + tn) / len(te))
    row = {
        "model": name,
        "recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
        "spec_mean": np.mean(specs),     "spec_std": np.std(specs),
        "acc_mean": np.mean(accs),       "acc_std": np.std(accs),
    }
    print(f"{name:<26} recall {row['recall_mean']:.4f} +/- {row['recall_std']:.4f} | "
          f"specificity {row['spec_mean']:.4f} +/- {row['spec_std']:.4f} | "
          f"accuracy {row['acc_mean']:.4f} +/- {row['acc_std']:.4f}")
    return row

print("\n=== 3. Baselines (5-fold x 3 repeats = 15 scores each) ===")
models = {
    "always says seizure":      lambda: DummyClassifier(strategy="most_frequent"),
    "logistic regression":      lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "one tree, depth 5":        lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
    "forest 10 trees, depth 5": lambda: RandomForestClassifier(n_estimators=10, max_depth=5,
                                                               class_weight="balanced", random_state=SEED),
}
results = [evaluate(name, make) for name, make in models.items()]

pd.DataFrame(results).to_csv("results_step1_baseline.csv", index=False)
print("\nSaved: results_step1_baseline.csv")