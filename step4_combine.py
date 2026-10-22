"""
step4_combine.py  --  Step 4: shrink the bit width, then combine every saving into one design.

The story: so far we found three separate savings.
    1. smaller model  (fewer trees / shallower trees)      -> fewer comparators   (Step 2, 3)
    2. fewer inputs   (6 of the 16 columns are enough)      -> fewer wires        (Step 3)
    3. narrower numbers (this step)                         -> smaller comparators
Your Verilog uses 16-bit numbers, but the data only needs 10 bits. Dropping low bits is the
same as using a cheaper, coarser ADC: each bit you drop halves the number of levels the
detector can see. We ask how coarse the ADC can get before the alarm stops being good enough.

Then we try every combination (6 models x 2 input sets x 7 resolutions = 84 designs), score
each one fairly, put a rough price on it, and pick the cheapest one that clears the bar.

PRICE (a rough proxy, not real gates):  comparator-bits = comparators x bit width.
A comparator that looks at a w-bit number needs roughly w units of logic, so this number
moves in the right direction. Step 5 replaces it with real synthesis numbers.

Run from C:\\Biomed_project inside the venv:   python step4_combine.py
(a few minutes; results are saved to results_step4_combined.csv and a picture)
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

SEED = 42
MIN_RECALL, MIN_SPEC = 0.99, 0.95          # the same "good enough" bar as before
BITS_IN_YOUR_VERILOG = 16                  # width used by seizure_detector.v today

df = pd.read_csv("BEED_Data.csv")
names = list(df.columns[:-1])
X_full = df.drop(columns=["y"]).to_numpy().astype(np.int64)
y = (df["y"] > 0).astype(int).to_numpy()

# The 6 inputs Step 3 found (X16, X8, X2, X7, X15, X11), as column positions.
SIX_INPUTS = [names.index(n) for n in ["X16", "X8", "X2", "X7", "X15", "X11"]]
INPUT_SETS = {"all 16": list(range(len(names))), "best 6": SIX_INPUTS}

# (label, kind, trees, depth)
DESIGNS = [("tree d7", "tree", 1, 7), ("tree d8", "tree", 1, 8), ("tree d10", "tree", 1, 10),
           ("forest 3x6", "forest", 3, 6), ("forest 5x5", "forest", 5, 5),
           ("forest 10x5 (yours)", "forest", 10, 5)]
SHIFTS = [0, 1, 2, 3, 4, 5, 6]             # how many low bits to drop

folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X_full, y))


def width_needed(values):
    """Signed bits needed to hold these whole numbers."""
    return math.ceil(math.log2(int(np.abs(values).max()) + 1)) + 1


def make_model(kind, trees, depth, n_inputs):
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)
    return RandomForestClassifier(n_estimators=trees, max_depth=depth,
                                  max_features=min(4, n_inputs),     # as in Steps 2 and 3
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def hardware_predict(model, kind, X_test):
    """One tree: its own answer. Forest: alert when MORE THAN HALF of the trees vote seizure."""
    if kind == "tree":
        return model.predict(X_test)
    votes = sum(t.predict(X_test).astype(int) for t in model.estimators_)
    return (votes >= len(model.estimators_) // 2 + 1).astype(int)


def count_comparators(model, kind):
    """Every decision (non-leaf) node is one comparator in hardware."""
    trees = [model] if kind == "tree" else model.estimators_
    return sum(t.tree_.node_count - t.tree_.n_leaves for t in trees)


# ------------------------------------------------------------------ the big sweep
rows = []
for set_name, cols in INPUT_SETS.items():
    for shift in SHIFTS:
        Xq = np.floor_divide(X_full[:, cols], 2 ** shift)   # drop `shift` low bits (floor, like a bit-shift)
        width = width_needed(Xq)
        for label, kind, trees, depth in DESIGNS:
            rec, spe, acc, cost = [], [], [], []
            for tr, te in folds:
                model = make_model(kind, trees, depth, len(cols))
                model.fit(Xq[tr], y[tr])
                pred = hardware_predict(model, kind, Xq[te])
                tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
                rec.append(tp / (tp + fn)); spe.append(tn / (tn + fp)); acc.append((tp + tn) / len(te))
                cost.append(count_comparators(model, kind))
            row = {"design": label, "inputs": set_name, "n_inputs": len(cols), "dropped_bits": shift,
                   "width": width, "comparators": np.mean(cost),
                   "recall": np.mean(rec), "spec": np.mean(spe), "acc": np.mean(acc)}
            row["comparator_bits"] = row["comparators"] * width
            row["input_bits"] = len(cols) * width
            row["good_enough"] = row["recall"] >= MIN_RECALL and row["spec"] >= MIN_SPEC
            rows.append(row)
            print(f"{label:<20} {set_name:<7} width={width:>2}  comparators={row['comparators']:6.1f}  "
                  f"recall={row['recall']:.4f}  spec={row['spec']:.4f}  "
                  f"cost={row['comparator_bits']:7.1f}  {'PASSES' if row['good_enough'] else ''}")

res = pd.DataFrame(rows)
res.to_csv("results_step4_combined.csv", index=False)

# ------------------------------------------------------------------ the answer
ref = res[(res.design == "forest 10x5 (yours)") & (res.inputs == "all 16") & (res.dropped_bits == 0)].iloc[0]
ref_cost = ref.comparators * BITS_IN_YOUR_VERILOG
print("\n=== Your Verilog today (10 trees, depth 5, all 16 inputs, 16-bit numbers) ===")
print(f"comparators={ref.comparators:.1f}  recall={ref.recall:.4f}  spec={ref.spec:.4f}  acc={ref.acc:.4f}  "
      f"cost={ref_cost:.0f} comparator-bits, input wires={16 * BITS_IN_YOUR_VERILOG} bits")

good = res[res.good_enough].sort_values(["comparator_bits", "input_bits"])
print(f"\n=== Ten cheapest designs with recall >= {MIN_RECALL} and specificity >= {MIN_SPEC} ===")
if good.empty:
    print("No design clears the bar.")
else:
    show = good.head(10)[["design", "inputs", "width", "comparators", "recall", "spec", "acc",
                          "comparator_bits", "input_bits"]]
    print(show.round(4).to_string(index=False))
    best = good.iloc[0]
    print(f"\nCheapest: {best.design}, {best.inputs} inputs, {int(best.width)}-bit numbers")
    print(f"  comparator cost {best.comparator_bits:.0f} vs {ref_cost:.0f} today  ->  "
          f"{ref_cost / best.comparator_bits:.1f}x smaller")
    print(f"  input wires     {int(best.input_bits)} bits vs {16 * BITS_IN_YOUR_VERILOG} today  ->  "
          f"{16 * BITS_IN_YOUR_VERILOG / best.input_bits:.1f}x smaller")
    print(f"  quality change: recall {best.recall - ref.recall:+.4f}, spec {best.spec - ref.spec:+.4f}, "
          f"accuracy {best.acc - ref.acc:+.4f}")

# ------------------------------------------------------------------ the picture
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, col, bar, label in [(axes[0], "recall", MIN_RECALL, "Seizures caught (recall)"),
                                (axes[1], "spec", MIN_SPEC, "Calm periods left alone (specificity)")]:
        ax.scatter(res.comparator_bits[~res.good_enough], res[col][~res.good_enough], s=18,
                   color="#8a96a0", label="misses the bar on either measure")
        ax.scatter(res.comparator_bits[res.good_enough], res[col][res.good_enough], s=18,
                   color="#0b7a70", label="clears the bar on both")
        ax.scatter([ref_cost], [ref[col]], marker="*", s=200, color="#c0392b", label="your Verilog today")
        if not good.empty:
            ax.scatter([best.comparator_bits], [best[col]], marker="D", s=70, facecolors="none",
                       edgecolors="#2c3e50", linewidths=2, label="cheapest that clears")
        ax.axhline(bar, color="#c0392b", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Cost: comparator-bits (comparators x bit width, log scale)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("84 designs: model size x inputs x number resolution")
    fig.tight_layout()
    fig.savefig("step4_combined.png", dpi=150)
    print("\nSaved: results_step4_combined.csv and step4_combined.png")
except ImportError:
    print("\nSaved: results_step4_combined.csv  (for the plot, run: pip install matplotlib)")