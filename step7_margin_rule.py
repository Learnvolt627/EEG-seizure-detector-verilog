"""
step7_margin_rule.py  --  Step 7a: choose with a safety margin.

The story: in Step 6 the rule "cheapest design that clears the bar" picked a design that
cleared it by a hair on development data and then failed on 11 of 20 new splits. Here we
repeat the same exam, but the rule now demands a SAFETY MARGIN when choosing:
development recall >= SELECT_RECALL and specificity >= SELECT_SPEC (stricter than the bar we
finally judge by). Everything else is as in Step 6:

  Part A  Lock away 25% of the rows (the HOLDOUT). Choose the design using only the other 75%
          (the DEVELOPMENT part), by cross-validation: cheapest design that clears the bar.
  Part B  Open the holdout ONCE. Score the chosen design, your current forest, and some
          reference designs, with error bars (95% confidence intervals).
  Part C  Repeat the 75/25 split with 20 different random seeds, to see how much the numbers
          wobble from luck of the split.

Reference designs in Part B tell the story:
  "tree d7, 16-bit"  same tree as 'robust' but full-width numbers -> separates the effect of
                     "one tree instead of ten" from "narrower numbers"
  SVM / small neural net / forest of 100 trees -> what software with no hardware limit achieves

The bar: seizure recall >= 0.99 and specificity >= 0.95 (as in earlier steps).

Run from C:\\Biomed_project inside the venv:   python step7_margin_rule.py
(a few minutes; saves results_step7_*.csv)
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

SEED = 42
MIN_RECALL, MIN_SPEC = 0.99, 0.95            # the bar we JUDGE by (unchanged)
SELECT_RECALL, SELECT_SPEC = 0.993, 0.96     # the stricter bar we CHOOSE by (the margin)

df = pd.read_csv("BEED_Data.csv")
X = df.drop(columns=["y"]).to_numpy().astype(np.int64)
y = (df["y"] > 0).astype(int).to_numpy()


def width_needed(values):
    return math.ceil(math.log2(int(np.abs(values).max()) + 1)) + 1


def quant(A, shift):
    """Drop `shift` low bits (floor), like a coarser ADC / a hardware right-shift."""
    return np.floor_divide(A, 2 ** shift)


def make_tree(depth):
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)


def make_forest(n, depth):
    return RandomForestClassifier(n_estimators=n, max_depth=depth, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def hard_vote(forest, A):
    votes = sum(t.predict(A).astype(int) for t in forest.estimators_)
    return (votes >= len(forest.estimators_) // 2 + 1).astype(int)


def comparators(model):
    ts = model.estimators_ if hasattr(model, "estimators_") else [model]
    return sum(t.tree_.node_count - t.tree_.n_leaves for t in ts)


def rates(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return tp, fn, tn, fp


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fit_predict(spec, Xtr, ytr, Xte):
    """spec = (kind, depth_or_trees..., shift). Returns predictions and comparators (None for software refs)."""
    kind = spec[0]
    if kind == "tree":
        _, depth, shift = spec
        m = make_tree(depth).fit(quant(Xtr, shift), ytr)
        return m.predict(quant(Xte, shift)), comparators(m)
    if kind == "forest":
        _, n, depth, shift = spec
        m = make_forest(n, depth).fit(quant(Xtr, shift), ytr)
        return hard_vote(m, quant(Xte, shift)), comparators(m)
    if kind == "svm":
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")).fit(Xtr, ytr)
        return m.predict(Xte), None
    if kind == "mlp":
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                                          random_state=SEED)).fit(Xtr, ytr)
        return m.predict(Xte), None
    raise ValueError(kind)


# ============================================================ split: development / holdout
idx = np.arange(len(y))
dev, hold = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)
print(f"development rows: {len(dev)}   holdout rows (locked away): {len(hold)}")

# ============================================================ Part A: choose on development only
print("\n=== Part A: choosing a design using ONLY the development rows ===")
DEPTHS, SHIFTS = [5, 6, 7, 8, 10], [0, 1, 2, 3, 4, 5]
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X[dev], y[dev]))
rows = []
for depth in DEPTHS:
    for shift in SHIFTS:
        Xd = quant(X[dev], shift)
        r, s, c = [], [], []
        for tr, te in cv:
            m = make_tree(depth).fit(Xd[tr], y[dev][tr])
            tp, fn, tn, fp = rates(y[dev][te], m.predict(Xd[te]))
            r.append(tp / (tp + fn)); s.append(tn / (tn + fp)); c.append(comparators(m))
        width = width_needed(Xd)
        rows.append({"depth": depth, "drop_bits": shift, "width": width, "comparators": np.mean(c),
                     "cost": np.mean(c) * width, "recall": np.mean(r), "spec": np.mean(s)})
grid = pd.DataFrame(rows)
grid["ok"] = (grid.recall >= SELECT_RECALL) & (grid.spec >= SELECT_SPEC)
grid.to_csv("results_step7_selection.csv", index=False)
print(grid.sort_values("cost").head(12).round(4).to_string(index=False))
if grid.ok.any():
    pick = grid[grid.ok].sort_values("cost").iloc[0]
    print("\nCheapest single tree that clears the MARGIN bar on development data:")
else:
    pick = grid.assign(bal=(grid.recall + grid.spec) / 2).sort_values("bal", ascending=False).iloc[0]
    print("\nNo single tree clears the margin bar on development data; taking the best balanced score:")
print(f"  depth={int(pick.depth)}, drop {int(pick.drop_bits)} low bits ({int(pick.width)}-bit numbers), "
      f"recall={pick.recall:.4f}, spec={pick.spec:.4f}")
chosen = ("tree", int(pick.depth), int(pick.drop_bits))

# ============================================================ Part B: open the holdout once
print("\n=== Part B: the holdout exam (opened once) ===")
CANDIDATES = [
    ("CHOSEN (picked in Part A)",     chosen),
    ("your forest 10x5, 16-bit",      ("forest", 10, 5, 0)),
    ("robust: tree d7, 6-bit",        ("tree", 7, 4)),
    ("aggressive: tree d8, 5-bit",    ("tree", 8, 5)),
    ("tree d7, 16-bit (reference)",   ("tree", 7, 0)),
    ("forest of 100, depth 10 (software)", ("forest", 100, 10, 0)),
    ("SVM (software only)",           ("svm",)),
    ("small neural net (software only)", ("mlp",)),
]
rows = []
for label, spec in CANDIDATES:
    pred, comp = fit_predict(spec, X[dev], y[dev], X[hold])
    tp, fn, tn, fp = rates(y[hold], pred)
    rl, rh = wilson(tp, tp + fn)
    sl, sh = wilson(tn, tn + fp)
    row = {"design": label, "comparators": comp, "recall": tp / (tp + fn), "recall_ci": f"{rl:.3f}-{rh:.3f}",
           "spec": tn / (tn + fp), "spec_ci": f"{sl:.3f}-{sh:.3f}", "acc": (tp + tn) / len(hold),
           "missed_seizures": fn, "false_alarms": fp}
    rows.append(row)
hold_df = pd.DataFrame(rows)
hold_df.to_csv("results_step7_holdout.csv", index=False)
print(hold_df.round(4).to_string(index=False))
print(f"(holdout has {int(y[hold].sum())} seizure rows and {int((1 - y[hold]).sum())} calm rows; "
      "the confidence intervals show how far each rate could plausibly be from the truth)")

# ============================================================ Part C: 20 different splits
print("\n=== Part C: does it hold across 20 different random 75/25 splits? ===")
FIXED = [c for c in CANDIDATES if c[0] != "CHOSEN (picked in Part A)"]
FIXED = [("chosen design (fixed)", chosen)] + FIXED
per = {label: [] for label, _ in FIXED}
for seed in range(20):
    a, b = train_test_split(idx, test_size=0.25, random_state=1000 + seed, stratify=y)
    for label, spec in FIXED:
        pred, _ = fit_predict(spec, X[a], y[a], X[b])
        tp, fn, tn, fp = rates(y[b], pred)
        per[label].append((tp / (tp + fn), tn / (tn + fp), (tp + tn) / len(b)))
rows = []
for label, vals in per.items():
    v = np.array(vals)
    rows.append({"design": label, "recall": f"{v[:, 0].mean():.4f} +/- {v[:, 0].std():.4f}",
                 "spec": f"{v[:, 1].mean():.4f} +/- {v[:, 1].std():.4f}",
                 "acc": f"{v[:, 2].mean():.4f} +/- {v[:, 2].std():.4f}",
                 "splits_clearing_bar": int(((v[:, 0] >= MIN_RECALL) & (v[:, 1] >= MIN_SPEC)).sum())})
seeds_df = pd.DataFrame(rows)
seeds_df.to_csv("results_step7_seeds.csv", index=False)
print(seeds_df.to_string(index=False))
print("(splits_clearing_bar = how many of the 20 splits met recall >= 0.99 AND specificity >= 0.95)")
print("\nSaved: results_step7_selection.csv, results_step7_holdout.csv, results_step7_seeds.csv")