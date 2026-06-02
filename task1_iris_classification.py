# ============================================================
# TASK 1: IRIS FLOWER CLASSIFICATION
# Internship - Data Science | InAmigos Foundation
# ============================================================

# Install required libraries (run once in terminal):
# pip install scikit-learn pandas numpy matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─── 1. LOAD DATASET ──────────────────────────────────────────
print("=" * 55)
print("   IRIS FLOWER CLASSIFICATION")
print("=" * 55)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\n[INFO] First 5 rows of dataset:")
print(df.head())
print(f"\n[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Species distribution:\n{df['species'].value_counts()}")

# ─── 2. EXPLORATORY DATA ANALYSIS ────────────────────────────
print("\n[INFO] Basic statistics:")
print(df.describe())

# Pairplot
sns.pairplot(df, hue='species', palette='husl', markers=['o', 's', 'D'])
plt.suptitle("Iris Dataset - Pairplot by Species", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("iris_pairplot.png", dpi=100, bbox_inches='tight')
plt.show()
print("[SAVED] iris_pairplot.png")

# Correlation Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_correlation.png", dpi=100)
plt.show()
print("[SAVED] iris_correlation.png")

# ─── 3. PREPARE DATA ─────────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n[INFO] Training samples: {X_train.shape[0]}")
print(f"[INFO] Testing  samples: {X_test.shape[0]}")

# ─── 4. TRAIN MULTIPLE MODELS ────────────────────────────────
models = {
    "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":         DecisionTreeClassifier(random_state=42),
    "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "-" * 55)
print("  MODEL COMPARISON")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:30s}  Accuracy: {acc * 100:.2f}%")

# ─── 5. BEST MODEL - DETAILED EVALUATION ─────────────────────
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

print(f"\n[BEST MODEL] {best_model_name}  ({results[best_model_name]*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=100)
plt.show()
print("[SAVED] iris_confusion_matrix.png")

# ─── 6. FEATURE IMPORTANCE (Random Forest) ───────────────────
rf = models["Random Forest"]
importances = rf.feature_importances_
feat_names = iris.feature_names

plt.figure(figsize=(7, 4))
bars = plt.barh(feat_names, importances, color=['#2196F3','#4CAF50','#FF9800','#E91E63'])
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=100)
plt.show()
print("[SAVED] iris_feature_importance.png")

# ─── 7. PREDICT NEW SAMPLE ───────────────────────────────────
print("\n" + "=" * 55)
print("  PREDICT A NEW FLOWER SAMPLE")
print("=" * 55)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical setosa
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)
print(f"Input features : sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"Predicted species: {iris.target_names[prediction[0]].upper()}")

print("\n[DONE] Task 1 complete!")
