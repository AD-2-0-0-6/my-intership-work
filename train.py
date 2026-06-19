import os
import joblib
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
df = pd.concat([X, pd.Series(y, name='target')], axis=1)

print('Features shape:', X.shape)
print('Feature names:', iris.feature_names)

# 2. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 3. Evaluate
y_pred = rf_classifier.predict(X_test)
acc_val = float(accuracy_score(y_test, y_pred))
print(f'\nRandomForest Test Accuracy: {acc_val:.4f}')

# 4. Save Artifacts to a dedicated directory
os.makedirs('models', exist_ok=True)

# Save serialized weights (Fixing variable references)
joblib.dump(rf_classifier, 'models/iris_model.joblib')
with open('models/iris_model.pickle', 'wb') as f:
    pickle.dump(rf_classifier, f)

# Save metadata schema
model_info = {
    'model_type': 'RandomForestClassifier',
    'accuracy': acc_val,
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist()
}
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

# Save feature boundary configurations (Completed structural breakdown)
feature_ranges = {
    'sepal_length': {
        'min': float(X['sepal length (cm)'].min()), 
        'max': float(X['sepal length (cm)'].max()),
        'default': float(X['sepal length (cm)'].mean())
    },
    'sepal_width': {
        'min': float(X['sepal width (cm)'].min()), 
        'max': float(X['sepal width (cm)'].max()),
        'default': float(X['sepal width (cm)'].mean())
    },
    'petal_length': {
        'min': float(X['petal length (cm)'].min()), 
        'max': float(X['petal length (cm)'].max()),
        'default': float(X['petal length (cm)'].mean())
    },
    'petal_width': {
        'min': float(X['petal width (cm)'].min()), 
        'max': float(X['petal width (cm)'].max()),
        'default': float(X['petal width (cm)'].mean())
    }
}
with open('models/feature_ranges.json', 'w') as f:
    json.dump(feature_ranges, f, indent=2)

print("\n✅ Saved: models/iris_model.joblib, models/iris_model.pickle, models/model_info.json, models/feature_ranges.json")