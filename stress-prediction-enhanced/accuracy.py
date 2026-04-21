import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

df = pd.read_csv('workplace_stress_dataset_4k_dramatically_improved.csv')

for col in df.select_dtypes(include='object').columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

target_col = 'calculated_stress_level'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("="*55)
print("      STRESSIQ — MODEL EVALUATION RESULTS")
print("="*55)
print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score  : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='macro'):.4f}")
print("="*55)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import cross_val_score
cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCV Mean: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")