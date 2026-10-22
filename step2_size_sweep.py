"""
step2_size_sweep.py  --  Step 2: how small can the forest get?

The story: you built one alarm (10 trees, depth 5). Now we build 36 alarms of different
sizes, test each one fairly, and count what each would cost in hardware. Then we ask
one question: what is the CHEAPEST alarm that is still good enough?

  - "size"  = how many trees, and how deep each tree may go
  - "cost"  = how many comparators the hardware needs (every decision node = 1 comparator)
  - "good"  = catches at least MIN_RECALL of seizures AND leaves at least MIN_SPEC
              of calm periods alone (change the two numbers below to move the bar)

Run from C:\\Biomed_project inside the venv:   python step2_size_sweep.py
(about 1-3 minutes; it prints one line per design as it goes)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

SEED = 42
TREES = [1, 2, 3, 5, 10, 20]          # how many trees to try
DEPTHS = [2, 3, 4, 5, 6, 8]           # how deep each tree may go
MIN_RECALL, MIN_SPEC = 0.99, 0.95     # the "good enough" bar

# ---------------------------------------------------------------- load the data
df = pd.read_csv("BEED_Data.csv")
X = df.drop(columns=["y"]).to_numpy()
y = (df["y"] > 0).astype(int).to_numpy()       # 0 = non-seizure, 1 = seizure

# Same 15 train/test splits for every design, so the comparison is fair.
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X, y))


def hardware_predict(forest, X_test):
    """Copy what the Verilog does: each tree casts a hard 0/1 vote, and the alert
    fires when MORE THAN HALF of the trees vote seizure (6 of 10 in your design)."""
    votes = np.zeros(len(X_test), dtype=int)
    for tree in forest.estimators_:
        votes += tree.predict(X_test).astype(int)
    need = len(forest.estimators_) // 2 + 1
    return (votes >= need).astype(int)


def comparators(forest):
    """Each decision (non-leaf) node becomes one comparator in hardware."""
    return sum(t.tree_.node_count - t.tree_.n_leaves for t in forest.estimators_)


# ---------------------------------------------------------------- the sweep
rows = []
for n_trees in TREES:
    for depth in DEPTHS:
        recalls, specs, accs, costs = [], [], [], []
        for tr, te in folds:
            forest = RandomForestClassifier(n_estimators=n_trees, max_depth=depth,
                                            class_weight="balanced",
                                            random_state=SEED, n_jobs=-1)
            forest.fit(X[tr], y[tr])
            pred = hardware_predict(forest, X[te])
            tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
            recalls.append(tp / (tp + fn))     # seizures caught
            specs.append(tn / (tn + fp))       # calm periods left alone
            accs.append((tp + tn) / len(te))
            costs.append(comparators(forest))
        row = {"trees": n_trees, "depth": depth,
               "comparators": float(np.mean(costs)),
               "recall": np.mean(recalls), "recall_std": np.std(recalls),
               "spec": np.mean(specs), "spec_std": np.std(specs),
               "acc": np.mean(accs)}
        row["good_enough"] = row["recall"] >= MIN_RECALL and row["spec"] >= MIN_SPEC
        rows.append(row)
        print(f"trees={n_trees:>2} depth={depth}  comparators={row['comparators']:6.1f}  "
              f"recall={row['recall']:.4f}  spec={row['spec']:.4f}  "
              f"{'PASSES' if row['good_enough'] else ''}")

res = pd.DataFrame(rows)
res.to_csv("results_step2_sweep.csv", index=False)

# ---------------------------------------------------------------- the answer
print("\n=== Your current design (10 trees, depth 5), scored the hardware way ===")
print(res[(res.trees == 10) & (res.depth == 5)][["comparators", "recall", "spec", "acc"]].round(4).to_string(index=False))

print(f"\n=== Cheapest designs with recall >= {MIN_RECALL} and specificity >= {MIN_SPEC} ===")
good = res[res.good_enough].sort_values("comparators")
if good.empty:
    print("No design clears the bar. Look at the plot and consider lowering MIN_RECALL / MIN_SPEC.")
else:
    print(good.head(8)[["trees", "depth", "comparators", "recall", "spec", "acc"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- the picture
try:
    import matplotlib
    matplotlib.use("Agg")                     # draw to a file, no window needed
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, col, bar, label in [(axes[0], "recall", MIN_RECALL, "Seizures caught (recall)"),
                                (axes[1], "spec", MIN_SPEC, "Calm periods left alone (specificity)")]:
        ax.scatter(res.comparators[~res.good_enough], res[col][~res.good_enough],
                   color="#8a96a0", label="below the bar")
        ax.scatter(res.comparators[res.good_enough], res[col][res.good_enough],
                   color="#0b7a70", label="clears the bar")
        for _, r in res[res.good_enough].iterrows():
            ax.annotate(f"{int(r.trees)}t/d{int(r.depth)}", (r.comparators, r[col]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(bar, color="#c0392b", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Cost: comparators in hardware (log scale)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle("Quality vs hardware cost for 36 forest sizes (labels: trees/depth)")
    fig.tight_layout()
    fig.savefig("step2_cost_vs_quality.png", dpi=150)
    print("\nSaved: results_step2_sweep.csv and step2_cost_vs_quality.png")
except ImportError:
    print("\nSaved: results_step2_sweep.csv  (for the plot, run: pip install matplotlib)")