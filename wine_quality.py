# ============================================================
# Wine Quality Classification
# Algorithms: Decision Tree & Random Forest
# Dataset: Red Wine Quality (UCI / Kaggle)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
DATA_PATH = os.path.join("data", "winequality-red.csv")
os.makedirs("images", exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")   # ফাইলে delimiter semicolon

print("=== Shape ===")
print(df.shape)
print("\n=== First 5 rows ===")
print(df.head())
print("\n=== Info ===")
print(df.info())
print("\n=== Missing values ===")
print(df.isnull().sum())
print("\n=== Statistical summary ===")
print(df.describe())

# ------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------
# 2.1 Quality distribution
plt.figure(figsize=(7, 4))
sns.countplot(x="quality", data=df, palette="viridis")
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/01_quality_distribution.png", dpi=150)
plt.close()

# 2.2 Correlation heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/02_correlation_heatmap.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 3. Create Target (Binary Classification)
#    quality >= 6  -> Good (1), otherwise Bad (0)
# ------------------------------------------------------------
df["good_quality"] = (df["quality"] >= 6).astype(int)
print("\n=== Class distribution ===")
print(df["good_quality"].value_counts())

X = df.drop(columns=["quality", "good_quality"])
y = df["good_quality"]

# ------------------------------------------------------------
# 4. Train / Test Split (80% / 20%)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ------------------------------------------------------------
# 5. Decision Tree
# ------------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_pred)
dt_cv = cross_val_score(dt, X, y, cv=5).mean()

print("\n================ DECISION TREE ================")
print(f"Train Accuracy : {dt_train_acc:.4f}")
print(f"Test Accuracy  : {dt_test_acc:.4f}")
print(f"5-Fold CV Acc  : {dt_cv:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, target_names=["Bad", "Good"]))

# Decision Tree visualization (depth 3 for readability)
plt.figure(figsize=(20, 10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["Bad", "Good"],
    filled=True,
    max_depth=3,
    fontsize=9,
)
plt.title("Decision Tree (top 3 levels)")
plt.tight_layout()
plt.savefig("images/03_decision_tree.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 6. Random Forest
# ------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_pred)
rf_cv = cross_val_score(rf, X, y, cv=5).mean()

print("\n================ RANDOM FOREST ================")
print(f"Train Accuracy : {rf_train_acc:.4f}")
print(f"Test Accuracy  : {rf_test_acc:.4f}")
print(f"5-Fold CV Acc  : {rf_cv:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=["Bad", "Good"]))

# ------------------------------------------------------------
# 7. Confusion Matrices
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, dt_pred), display_labels=["Bad", "Good"]
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Decision Tree")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, rf_pred), display_labels=["Bad", "Good"]
).plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.savefig("images/04_confusion_matrices.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 8. Feature Importance (Random Forest)
# ------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="teal")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/05_feature_importance.png", dpi=150)
plt.close()

print("\n=== Top Features (Random Forest) ===")
print(importances.sort_values(ascending=False).head(5))

# ------------------------------------------------------------
# 9. Model Comparison
# ------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Test Accuracy": [dt_test_acc, rf_test_acc],
    "CV Accuracy": [dt_cv, rf_cv],
})
print("\n=== Model Comparison ===")
print(results)

results.set_index("Model").plot(kind="bar", figsize=(7, 5), rot=0)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("images/06_model_comparison.png", dpi=150)
plt.close()

print("\nDone! All plots are saved in the 'images' folder.")
