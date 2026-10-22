"""
step3_inputs_and_checks.py  --  Step 3: how few inputs does the detector need?

The story: every input to your detector is a wire, a register and a comparator, and if the
16 inputs are electrodes, each one is also a sensor on someone's head. So we ask: which
inputs matter most, and how few can we keep before the alarm stops being good enough?

  Part A  What ARE X1..X16?  (a correlation check, so we word the result honestly)
  Part B  Add inputs one at a time, best first, and score the alarm after each addition
  Part C  Two small checks left over from Step 2:
            C1  is a plain decision tree better than the "forest of 1 tree"?
            C2  what does the vote rule (how many trees must agree) do?

Run from C:\\Biomed_project inside the venv:   python step3_inputs_and_checks.py
(a few minutes; results are also saved as CSV files)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

SEED = 42
MIN_RECALL, MIN_SPEC = 0.99, 0.95          # the same "good enough" bar as Step 2

df = pd.read_csv("BEED_Data.csv")
names = list(df.columns[:-1])              # X1 ... X16
X = df.drop(columns=["y"]).to_numpy()
y = (df["y"] > 0).astype(int).to_numpy()   # 0 = non-seizure, 1 = seizure

# 15 splits to SCORE designs, and a different set of 5 splits to CHOOSE inputs, so the
# choosing and the final scoring do not look at the data in exactly the same way.
score_folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X, y))
pick_folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED + 1).split(X, y))


def rates(y_true, pred):
    """Seizures caught (recall), calm periods left alone (specificity), and accuracy."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return tp / (tp + fn), tn / (tn + fp), (tp + tn) / len(y_true)


def votes_for(forest, X_test):
    """How many trees vote 'seizure' for each row (each tree votes 0 or 1, like the Verilog)."""
    return sum(tree.predict(X_test).astype(int) for tree in forest.estimators_)


def hardware_predict(forest, X_test):
    """Alert when MORE THAN HALF of the trees vote seizure."""
    return (votes_for(forest, X_test) >= len(forest.estimators_) // 2 + 1).astype(int)


# ============================================================ Part A: what are X1..X16?
print("=== Part A: how do the 16 columns relate to each other? ===")
corr = np.corrcoef(X.T)
print("lag = distance between column numbers (lag 1 = X1 vs X2, X2 vs X3, ...)")
print(f"{'lag':>4} {'mean corr':>10} {'min':>8} {'max':>8}")
for lag in range(1, 16):
    pairs = np.diag(corr, k=lag)
    print(f"{lag:>4} {pairs.mean():>10.3f} {pairs.min():>8.3f} {pairs.max():>8.3f}")
print("How to read this: if the columns are CONSECUTIVE SAMPLES of one signal, correlation is")
print("high at lag 1, falls smoothly as lag grows, and min/max stay close together at each lag.")
print("If they are SEPARATE ELECTRODES, correlation jumps around from pair to pair (wide")
print("min-to-max range) and does not fall steadily with lag. This is evidence, not proof.")

# ============================================================ Part B: inputs, best first
print("\n=== Part B: adding inputs one at a time (greedy forward selection) ===")


def forest_score(cols, folds, n_trees=5, depth=5):
    """Average recall / specificity / accuracy of a forest that may ONLY look at columns `cols`.
    max_features=min(4, ...) keeps the same '4 inputs considered per split' as Step 2, but
    never asks for more inputs than we have allowed."""
    out = []
    for tr, te in folds:
        forest = RandomForestClassifier(n_estimators=n_trees, max_depth=depth,
                                        max_features=min(4, len(cols)),
                                        class_weight="balanced", random_state=SEED, n_jobs=-1)
        forest.fit(X[tr][:, cols], y[tr])
        out.append(rates(y[te], hardware_predict(forest, X[te][:, cols])))
    return np.mean(out, axis=0)            # (recall, specificity, accuracy)


chosen, remaining = [], list(range(X.shape[1]))
while remaining:
    best_col, best_value = None, -1.0
    for col in remaining:
        recall, spec, _ = forest_score(chosen + [col], pick_folds)
        value = (recall + spec) / 2        # balanced accuracy: both numbers matter
        if value > best_value:
            best_col, best_value = col, value
    chosen.append(best_col)
    remaining.remove(best_col)
    print(f"  input #{len(chosen):>2} added: {names[best_col]:<4} (balanced accuracy while choosing: {best_value:.4f})")

print("\nNow scoring each prefix on the 15 scoring splits (forest of 5 trees, depth 5):")
rows = []
for k in range(1, len(chosen) + 1):
    recall, spec, acc = forest_score(chosen[:k], score_folds)
    ok = recall >= MIN_RECALL and spec >= MIN_SPEC
    rows.append({"inputs": k, "added": names[chosen[k - 1]], "recall": recall, "spec": spec,
                 "acc": acc, "good_enough": ok})
    print(f"  {k:>2} inputs (+{names[chosen[k - 1]]:<4}) recall={recall:.4f}  spec={spec:.4f}  "
          f"acc={acc:.4f}  {'PASSES' if ok else ''}")
inputs_df = pd.DataFrame(rows)
inputs_df.to_csv("results_step3_inputs.csv", index=False)

passing = inputs_df[inputs_df.good_enough]
if passing.empty:
    print("\nNo input count clears the bar with this design.")
else:
    k = int(passing.inputs.min())
    print(f"\nFewest inputs that clear the bar: {k}  ->  {', '.join(names[c] for c in chosen[:k])}")

# ============================================================ Part C1: plain tree
print("\n=== Part C1: one plain decision tree (no resampling, sees all inputs) ===")
rows = []
for depth in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
    out, cost = [], []
    for tr, te in score_folds:
        tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)
        tree.fit(X[tr], y[tr])
        out.append(rates(y[te], tree.predict(X[te])))
        cost.append(tree.tree_.node_count - tree.tree_.n_leaves)
    recall, spec, acc = np.mean(out, axis=0)
    ok = recall >= MIN_RECALL and spec >= MIN_SPEC
    rows.append({"depth": depth, "comparators": np.mean(cost), "recall": recall, "spec": spec,
                 "acc": acc, "good_enough": ok})
    print(f"  depth={depth:>2}  comparators={np.mean(cost):6.1f}  recall={recall:.4f}  "
          f"spec={spec:.4f}  acc={acc:.4f}  {'PASSES' if ok else ''}")
pd.DataFrame(rows).to_csv("results_step3_plain_tree.csv", index=False)
print("  (Step 2's 'forest of 1 tree, depth 8' had 45.9 comparators, recall 0.9914, spec 0.9658)")

# ============================================================ Part C2: the vote rule
print("\n=== Part C2: the vote rule (alert when at least k trees say seizure) ===")
rows = []
for n_trees, depth in [(10, 5), (5, 5), (3, 6)]:
    per_k = {k: [] for k in range(1, n_trees + 1)}
    for tr, te in score_folds:
        forest = RandomForestClassifier(n_estimators=n_trees, max_depth=depth,
                                        class_weight="balanced", random_state=SEED, n_jobs=-1)
        forest.fit(X[tr], y[tr])
        votes = votes_for(forest, X[te])
        for k in per_k:
            per_k[k].append(rates(y[te], (votes >= k).astype(int)))
    print(f"  {n_trees} trees, depth {depth}   (your hardware rule uses k = {n_trees // 2 + 1})")
    for k, vals in per_k.items():
        recall, spec, acc = np.mean(vals, axis=0)
        ok = recall >= MIN_RECALL and spec >= MIN_SPEC
        rows.append({"trees": n_trees, "depth": depth, "k": k, "recall": recall, "spec": spec,
                     "acc": acc, "good_enough": ok})
        print(f"    k={k:>2}  recall={recall:.4f}  spec={spec:.4f}  acc={acc:.4f}  {'PASSES' if ok else ''}")
pd.DataFrame(rows).to_csv("results_step3_votes.csv", index=False)

# ============================================================ the picture (Part B)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(inputs_df.inputs, inputs_df.recall, marker="o", color="#0b7a70", label="seizures caught (recall)")
    ax.plot(inputs_df.inputs, inputs_df.spec, marker="s", color="#2c3e50", label="calm periods left alone (specificity)")
    ax.axhline(MIN_RECALL, color="#0b7a70", linestyle="--", linewidth=1)
    ax.axhline(MIN_SPEC, color="#2c3e50", linestyle="--", linewidth=1)
    for _, r in inputs_df.iterrows():
        ax.annotate(r.added, (r.inputs, r.spec), fontsize=7, xytext=(0, -12), textcoords="offset points", ha="center")
    ax.set_xlabel("Number of inputs allowed (best first; label = input added at that step)")
    ax.set_ylabel("Score (dashed lines = the bar)")
    ax.set_title("Alarm quality vs number of inputs (5 trees, depth 5)")
    ax.set_xticks(range(1, len(chosen) + 1))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("step3_inputs_curve.png", dpi=150)
    print("\nSaved: results_step3_*.csv and step3_inputs_curve.png")
except ImportError:
    print("\nSaved: results_step3_*.csv  (for the plot, run: pip install matplotlib)")