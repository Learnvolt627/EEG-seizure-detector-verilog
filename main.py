import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('BEED_Data.csv')
X = df.drop(columns=['y'])
y = (df['y'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest with balanced class weighting
rf = RandomForestClassifier(n_estimators=10, max_depth=5, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Evaluate ML Accuracy & Recall
y_pred = rf.predict(X_test)
print("--- MODEL ACCURACY & EVALUATION REPORT ---")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))