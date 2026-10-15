import pandas as pd
import math
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree

df = pd.read_csv('BEED_Data.csv')
X = df.drop(columns=['y'])
y = (df['y'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=10, max_depth=5, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

def tree_to_json(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = math.floor(tree_.threshold[node])
            left = recurse(tree_.children_left[node])
            right = recurse(tree_.children_right[node])
            return {
                "type": "node",
                "feature": name,
                "threshold": threshold,
                "left": left,
                "right": right
            }
        else:
            value = tree_.value[node][0]
            leaf_class = int(np.argmax(value))
            return {
                "type": "leaf",
                "vote": leaf_class
            }

    return recurse(0)

forest_data = []
for estimator in model.estimators_:
    forest_data.append(tree_to_json(estimator, X.columns.tolist()))

with open('tree_rules.json', 'w') as f:
    json.dump(forest_data, f, indent=4)

print("SUCCESS: Extracted rules from 10 decision trees into 'tree_rules.json'.")