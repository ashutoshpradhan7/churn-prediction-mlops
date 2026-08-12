"""
notebooks/eda.py
----------------
Exploratory Data Analysis for the Telco Customer Churn Dataset.
Run this script to generate a full EDA report saved to reports/eda/

Covers:
  1. Dataset Overview
  2. Missing Values & Data Types
  3. Target Distribution (Class Imbalance)
  4. Numerical Feature Analysis
  5. Categorical Feature Analysis
  6. Correlation Analysis
  7. Feature vs Churn Relationships
  8. Key Insights Summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/raw/telco_churn.csv"
REPORT_DIR  = "reports/eda"
os.makedirs(REPORT_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
COLORS = [BLUE, RED, GREEN, "#F59E0B", "#8B5CF6", "#06B6D4"]


# ── 1. Load Data ──────────────────────────────────────────────────────────────

print("=" * 60)
print("TELCO CHURN — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df_clean = df.dropna(subset=["TotalCharges"]).copy()
df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

print(f"   After cleaning : {df_clean.shape[0]} rows")
print(f"   Dropped rows   : {df.shape[0] - df_clean.shape[0]} (blank TotalCharges)")


# ── 2. Dataset Overview ───────────────────────────────────────────────────────

print("\n📋 COLUMN OVERVIEW")
print("-" * 60)
overview = pd.DataFrame({
    "dtype":    df.dtypes,
    "nulls":    df.isnull().sum(),
    "null_%":   (df.isnull().sum() / len(df) * 100).round(2),
    "unique":   df.nunique(),
    "sample":   df.iloc[0]
})
print(overview.to_string())


# ── 3. Target Distribution ────────────────────────────────────────────────────

churn_counts = df_clean["Churn"].value_counts()
churn_pct    = df_clean["Churn"].value_counts(normalize=True) * 100

print(f"\n🎯 TARGET DISTRIBUTION")
print(f"   No Churn : {churn_counts[0]:,} ({churn_pct[0]:.1f}%)")
print(f"   Churn    : {churn_counts[1]:,} ({churn_pct[1]:.1f}%)")
print(f"   → Class imbalance ratio: {churn_counts[0]/churn_counts[1]:.1f}:1")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Target Variable — Churn Distribution", fontsize=14, fontweight="bold")

# Bar chart
axes[0].bar(["No Churn", "Churn"], churn_counts.values,
            color=[GREEN, RED], alpha=0.85, edgecolor="white", linewidth=1.5)
axes[0].set_ylabel("Count")
axes[0].set_title("Absolute Count")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

# Pie chart
axes[1].pie(churn_counts.values, labels=["No Churn", "Churn"],
            colors=[GREEN, RED], autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Percentage Split")

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: 01_target_distribution.png")


# ── 4. Numerical Features ─────────────────────────────────────────────────────

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

print(f"\n📈 NUMERICAL FEATURES")
print(df_clean[num_cols].describe().round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Numerical Features — Distribution & Churn Relationship",
             fontsize=14, fontweight="bold")

for i, col in enumerate(num_cols):
    # Histogram
    ax = axes[0, i]
    churn_no  = df_clean[df_clean["Churn"] == 0][col]
    churn_yes = df_clean[df_clean["Churn"] == 1][col]
    ax.hist(churn_no,  bins=30, alpha=0.6, color=GREEN, label="No Churn", density=True)
    ax.hist(churn_yes, bins=30, alpha=0.6, color=RED,   label="Churn",    density=True)
    ax.set_title(f"{col} — Distribution by Churn")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()

    # Boxplot
    ax2 = axes[1, i]
    data = [churn_no.values, churn_yes.values]
    bp = ax2.boxplot(data, patch_artist=True, labels=["No Churn", "Churn"],
                     medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor(GREEN)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(RED)
    bp["boxes"][1].set_alpha(0.7)
    ax2.set_title(f"{col} — Boxplot by Churn")
    ax2.set_ylabel(col)

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/02_numerical_features.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: 02_numerical_features.png")

# Key stats
for col in num_cols:
    mean_churn    = df_clean[df_clean["Churn"] == 1][col].mean()
    mean_no_churn = df_clean[df_clean["Churn"] == 0][col].mean()
    print(f"   {col:<20} Churn mean: {mean_churn:.1f}  |  No Churn mean: {mean_no_churn:.1f}")


# ── 5. Categorical Features ───────────────────────────────────────────────────

cat_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

print(f"\n📊 CATEGORICAL FEATURES — Churn Rate per Category")
print("-" * 60)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Categorical Features — Churn Rate", fontsize=14, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_rate = df_clean.groupby(col)["Churn"].mean() * 100
    churn_rate = churn_rate.sort_values(ascending=False)

    bars = axes[i].bar(range(len(churn_rate)), churn_rate.values,
                       color=COLORS[:len(churn_rate)], alpha=0.85, edgecolor="white")
    axes[i].set_xticks(range(len(churn_rate)))
    axes[i].set_xticklabels(churn_rate.index, rotation=20, ha="right", fontsize=9)
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_ylabel("Churn Rate (%)")
    axes[i].axhline(y=churn_pct[1], color="black", linestyle="--",
                    linewidth=1, alpha=0.5, label=f"Avg {churn_pct[1]:.1f}%")
    axes[i].legend(fontsize=8)

    for bar, val in zip(bars, churn_rate.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    print(f"   {col}:")
    for cat, rate in churn_rate.items():
        print(f"      {str(cat):<35} {rate:.1f}%")

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/03_categorical_features.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   ✅ Saved: 03_categorical_features.png")


# ── 6. Contract Type Deep Dive ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Contract Type — Key Churn Driver", fontsize=14, fontweight="bold")

contract_churn = df_clean.groupby("Contract")["Churn"].agg(["mean", "count"])
contract_churn["mean"] *= 100

axes[0].bar(contract_churn.index, contract_churn["mean"],
            color=[RED, "#F59E0B", GREEN], alpha=0.85, edgecolor="white", linewidth=1.5)
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_title("Churn Rate by Contract Type")
for i, (idx, row) in enumerate(contract_churn.iterrows()):
    axes[0].text(i, row["mean"] + 0.5, f"{row['mean']:.1f}%",
                 ha="center", fontweight="bold")

tenure_bins = pd.cut(df_clean["tenure"], bins=[0,12,24,48,72], labels=["0-1yr","1-2yr","2-4yr","4+yr"])
tenure_churn = df_clean.groupby([tenure_bins, "Contract"])["Churn"].mean().unstack() * 100
tenure_churn.plot(kind="bar", ax=axes[1], color=[RED, "#F59E0B", GREEN],
                  alpha=0.85, edgecolor="white", linewidth=0.5)
axes[1].set_title("Churn Rate: Tenure Group × Contract")
axes[1].set_xlabel("Tenure Group")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].legend(title="Contract", bbox_to_anchor=(1.01, 1))
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/04_contract_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   ✅ Saved: 04_contract_analysis.png")


# ── 7. Correlation Heatmap ────────────────────────────────────────────────────

# Encode all categoricals for correlation
df_enc = df_clean.copy()
for col in df_enc.select_dtypes(include="object").columns:
    df_enc[col] = pd.Categorical(df_enc[col]).codes

corr = df_enc.drop(columns=["customerID"], errors="ignore").corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 8}, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: 05_correlation_heatmap.png")

# Top correlations with Churn
churn_corr = corr["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
print(f"\n🔗 TOP CORRELATIONS WITH CHURN:")
for feat, val in churn_corr.head(10).items():
    direction = "↑ increases" if val > 0 else "↓ decreases"
    print(f"   {feat:<25} {val:+.3f}  ({direction} churn risk)")


# ── 8. Monthly Charges vs Tenure Scatter ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))
scatter_no    = df_clean[df_clean["Churn"] == 0]
scatter_churn = df_clean[df_clean["Churn"] == 1]

ax.scatter(scatter_no["tenure"],    scatter_no["MonthlyCharges"],
           alpha=0.3, c=GREEN, s=15, label="No Churn")
ax.scatter(scatter_churn["tenure"], scatter_churn["MonthlyCharges"],
           alpha=0.5, c=RED,   s=15, label="Churn")
ax.set_xlabel("Tenure (months)", fontsize=12)
ax.set_ylabel("Monthly Charges ($)", fontsize=12)
ax.set_title("Tenure vs Monthly Charges — Churn Pattern", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/06_tenure_vs_charges.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: 06_tenure_vs_charges.png")


# ── 9. Internet Service & Add-ons ─────────────────────────────────────────────

service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

churn_rates = {}
for col in service_cols:
    rate = df_clean[df_clean[col] == "Yes"]["Churn"].mean() * 100
    no_rate = df_clean[df_clean[col] == "No"]["Churn"].mean() * 100
    churn_rates[col] = {"Has Service": rate, "No Service": no_rate}

churn_service_df = pd.DataFrame(churn_rates).T

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(churn_service_df))
width = 0.35
ax.bar(x - width/2, churn_service_df["Has Service"], width,
       label="Has Service", color=GREEN, alpha=0.85, edgecolor="white")
ax.bar(x + width/2, churn_service_df["No Service"], width,
       label="No Service", color=RED, alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(churn_service_df.index, rotation=20, ha="right")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Add-on Services — Churn Rate: Has Service vs No Service",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/07_service_addons.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved: 07_service_addons.png")


# ── 10. Key Insights Summary ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("💡 KEY EDA INSIGHTS — JUSTIFYING PREPROCESSING DECISIONS")
print("=" * 60)

insights = [
    ("Class Imbalance",
     f"73.5% No Churn vs 26.5% Churn → Applied SMOTE to balance training data"),
    ("Contract Type",
     "Month-to-month customers churn at ~43% vs 11% (1yr) and 3% (2yr) → Strong feature"),
    ("Tenure",
     "Churners have low tenure (avg ~18 months) vs stayers (avg ~37 months) → TenureGroup feature created"),
    ("Monthly Charges",
     "Churners pay higher monthly charges (~$74) vs stayers (~$61) → ChargePerTenure feature created"),
    ("Internet Service",
     "Fiber optic customers churn at 42% vs DSL 19% → High predictive power"),
    ("Add-on Services",
     "Customers WITHOUT security/backup/support churn 2x more → NumServices feature created"),
    ("Payment Method",
     "Electronic check users churn at 45% — highest of all payment methods"),
    ("Senior Citizens",
     "Senior customers churn at 41% vs 24% for non-seniors → Kept as feature"),
    ("TotalCharges",
     "11 rows had blank TotalCharges (new customers, tenure=0) → Dropped safely"),
    ("Correlations",
     "TotalCharges highly correlated with tenure (0.83) → Feature engineering handles multicollinearity"),
]

for i, (title, detail) in enumerate(insights, 1):
    print(f"\n  {i:02d}. {title}")
    print(f"      → {detail}")


# ── 11. Save HTML Summary ─────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>EDA Report — Telco Churn</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1e40af; border-bottom: 3px solid #1e40af; padding-bottom: 10px; }}
        h2 {{ color: #374151; margin-top: 30px; }}
        img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; margin: 10px 0 20px; }}
        .insight {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px;
                    margin: 8px 0; border-radius: 0 4px 4px 0; }}
        .insight strong {{ color: #1e40af; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
        .stat {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
                 padding: 14px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #1e40af; }}
        .stat-label {{ color: #6b7280; font-size: 0.85em; margin-top: 4px; }}
    </style>
</head>
<body>
    <h1>📊 EDA Report — Telco Customer Churn</h1>

    <div class="stat-grid">
        <div class="stat"><div class="stat-value">7,032</div><div class="stat-label">Total Customers</div></div>
        <div class="stat"><div class="stat-value">26.5%</div><div class="stat-label">Overall Churn Rate</div></div>
        <div class="stat"><div class="stat-value">20</div><div class="stat-label">Features Analyzed</div></div>
    </div>

    <h2>1. Target Distribution</h2>
    <img src="01_target_distribution.png">

    <h2>2. Numerical Features</h2>
    <img src="02_numerical_features.png">

    <h2>3. Categorical Features</h2>
    <img src="03_categorical_features.png">

    <h2>4. Contract Type Analysis</h2>
    <img src="04_contract_analysis.png">

    <h2>5. Correlation Heatmap</h2>
    <img src="05_correlation_heatmap.png">

    <h2>6. Tenure vs Monthly Charges</h2>
    <img src="06_tenure_vs_charges.png">

    <h2>7. Add-on Services Impact</h2>
    <img src="07_service_addons.png">

    <h2>💡 Key Insights & Preprocessing Decisions</h2>
    {"".join(f'<div class="insight"><strong>{t}</strong> → {d}</div>' for t, d in insights)}
</body>
</html>"""

with open(f"{REPORT_DIR}/eda_report.html", "w") as f:
    f.write(html)

print(f"\n✅ HTML EDA report saved to {REPORT_DIR}/eda_report.html")
print("\n🎉 EDA Complete! Open reports/eda/eda_report.html in your browser.")
print("=" * 60)
