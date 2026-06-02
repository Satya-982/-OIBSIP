# ============================================================
# TASK 5: SALES PREDICTION USING PYTHON
# Internship - Data Science | InAmigos Foundation
# ============================================================

# Install required libraries (run once in terminal):
# pip install scikit-learn pandas numpy matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── 1. LOAD / CREATE DATASET ─────────────────────────────────
# Dataset: Advertising dataset (TV, Radio, Newspaper budgets → Sales)
# Download from: https://www.kaggle.com/datasets/bumba5341/advertisingcsv
# OR use the built-in URL below.

print("=" * 55)
print("   SALES PREDICTION USING PYTHON")
print("=" * 55)

import os, urllib.request

DATA_FILE = "advertising.csv"
DATA_URL  = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

if not os.path.exists(DATA_FILE):
    print("[INFO] Downloading Advertising dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("[INFO] Download successful.")
    except Exception as e:
        print(f"[WARNING] Download failed: {e}")
        print("[INFO] Generating synthetic advertising dataset...")
        np.random.seed(42)
        n = 200
        TV        = np.random.uniform(0.7, 296, n)
        Radio     = np.random.uniform(0.0, 49.6, n)
        Newspaper = np.random.uniform(0.3, 114, n)
        # Sales = function of TV + Radio + noise
        Sales = 2.9 + 0.046 * TV + 0.188 * Radio + 0.001 * Newspaper + np.random.normal(0, 1.0, n)
        df_synth = pd.DataFrame({'TV': TV, 'Radio': Radio, 'Newspaper': Newspaper, 'Sales': Sales})
        df_synth.to_csv(DATA_FILE, index=False)
        print("[INFO] Synthetic dataset saved.")

df = pd.read_csv(DATA_FILE)
# Drop index column if present
if df.columns[0] in ['Unnamed: 0', '']:
    df = df.drop(df.columns[0], axis=1)

print(f"\n[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")
print("\n[INFO] First 5 rows:")
print(df.head())
print("\n[INFO] Summary statistics:")
print(df.describe().round(2))

# ─── 2. EXPLORATORY DATA ANALYSIS ────────────────────────────
# Check missing values
print(f"\n[INFO] Missing values:\n{df.isnull().sum()}")

# Distribution of Sales
plt.figure(figsize=(6, 4))
df['Sales'].hist(bins=25, color='steelblue', edgecolor='white')
plt.title("Sales Distribution")
plt.xlabel("Sales (in thousands)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("sales_distribution.png", dpi=100)
plt.show()
print("[SAVED] sales_distribution.png")

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("sales_correlation.png", dpi=100)
plt.show()
print("[SAVED] sales_correlation.png")

# Scatter plots: each feature vs Sales
features = [c for c in df.columns if c != 'Sales']
fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
if len(features) == 1:
    axes = [axes]
colors = ['#2196F3', '#4CAF50', '#FF9800']
for ax, feat, color in zip(axes, features, colors):
    ax.scatter(df[feat], df['Sales'], alpha=0.5, color=color, edgecolors='none')
    ax.set_xlabel(feat)
    ax.set_ylabel("Sales")
    ax.set_title(f"{feat} vs Sales")
plt.suptitle("Advertising Budget vs Sales", fontsize=13)
plt.tight_layout()
plt.savefig("sales_scatter_plots.png", dpi=100)
plt.show()
print("[SAVED] sales_scatter_plots.png")

# ─── 3. PREPARE DATA ─────────────────────────────────────────
X = df[features]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ─── 4. TRAIN MODELS ─────────────────────────────────────────
models = {
    "Linear Regression":     LinearRegression(),
    "Ridge Regression":      Ridge(alpha=1.0),
    "Random Forest":         RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []
print("\n" + "-" * 65)
print(f"  {'Model':<25}  {'MAE':>7}  {'RMSE':>7}  {'R² Score':>9}")
print("-" * 65)

trained_models = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    trained_models[name] = (model, y_pred)
    print(f"  {name:<25}  {mae:>7.3f}  {rmse:>7.3f}  {r2:>9.4f}")

# ─── 5. BEST MODEL ───────────────────────────────────────────
results_df  = pd.DataFrame(results).sort_values('R2', ascending=False)
best_name   = results_df.iloc[0]['Model']
best_model, y_pred_best = trained_models[best_name]

print(f"\n[BEST MODEL] {best_name}  (R² = {results_df.iloc[0]['R2']:.4f})")

# Actual vs Predicted plot
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_best, alpha=0.7, color='darkorange', edgecolors='none')
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales - {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("sales_actual_vs_predicted.png", dpi=100)
plt.show()
print("[SAVED] sales_actual_vs_predicted.png")

# Residual plot
residuals = y_test.values - y_pred_best
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_best, residuals, alpha=0.6, color='steelblue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("sales_residuals.png", dpi=100)
plt.show()
print("[SAVED] sales_residuals.png")

# Feature importance / coefficients
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_)
else:
    importances = None

if importances is not None:
    plt.figure(figsize=(6, 4))
    plt.barh(features, importances, color=['#2196F3', '#4CAF50', '#FF9800'][:len(features)])
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig("sales_feature_importance.png", dpi=100)
    plt.show()
    print("[SAVED] sales_feature_importance.png")

# ─── 6. MODEL COMPARISON BAR CHART ───────────────────────────
plt.figure(figsize=(8, 4))
colors = ['#4CAF50' if n == best_name else '#90CAF9' for n in results_df['Model']]
plt.barh(results_df['Model'], results_df['R2'], color=colors)
plt.xlabel("R² Score")
plt.title("Model Comparison - R² Score")
plt.xlim(0, 1.05)
for i, v in enumerate(results_df['R2']):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig("sales_model_comparison.png", dpi=100)
plt.show()
print("[SAVED] sales_model_comparison.png")

# ─── 7. PREDICT NEW SALES ────────────────────────────────────
print("\n" + "=" * 55)
print("  PREDICT SALES FOR NEW BUDGETS")
print("=" * 55)

new_budgets = pd.DataFrame({
    'TV':        [200, 50,  150],
    'Radio':     [40,  10,  30],
    'Newspaper': [30,  5,   20],
})

new_scaled    = scaler.transform(new_budgets)
new_preds     = best_model.predict(new_scaled)

for i, row in new_budgets.iterrows():
    print(f"TV=${row['TV']:>5.0f}K  Radio=${row['Radio']:>4.0f}K  "
          f"Newspaper=${row['Newspaper']:>4.0f}K  →  "
          f"Predicted Sales: {new_preds[i]:.2f}K units")

print("\n[DONE] Task 5 complete!")
