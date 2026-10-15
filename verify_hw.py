import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# --- Same data split and model as main.py / extract_rules.py ---
df = pd.read_csv('BEED_Data.csv')
X = df.drop(columns=['y'])
y = (df['y'] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=10, max_depth=5,
                               class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

def hw_tree_vote(tree, x, mode):
    """Walk one tree the way the Verilog does: integer threshold, 'x > thr' goes right."""
    t = tree.tree_
    node = 0
    while t.children_left[node] != -1:                 # -1 means leaf
        thr = int(round(float(t.threshold[node]))) if mode == 'round' \
              else math.floor(t.threshold[node])
        feat = t.feature[node]
        node = t.children_right[node] if x[feat] > thr else t.children_left[node]
    return int(np.argmax(t.value[node][0]))            # leaf vote, same as extract_rules.py

def hw_predict(mode):
    """Hardware behaviour: 10 hard votes, alert if >= 6 votes."""
    preds = []
    for x in X_test.values:
        votes = sum(hw_tree_vote(tr, x, mode) for tr in model.estimators_)
        preds.append(1 if votes >= 6 else 0)
    return np.array(preds)

sk_pred    = model.predict(X_test)        # software reference
hw_round   = hw_predict('round')          # what your Verilog does now
hw_floor   = hw_predict('floor')          # proposed fix
truth      = y_test.values

print("Samples tested:", len(truth))
print("HW (round) vs sklearn mismatches:", int((hw_round != sk_pred).sum()))
print("HW (floor) vs sklearn mismatches:", int((hw_floor != sk_pred).sum()))
print("\nHardware (current) vs ground truth:\n", confusion_matrix(truth, hw_round))
print("\nHardware (floor fix) vs ground truth:\n", confusion_matrix(truth, hw_floor))