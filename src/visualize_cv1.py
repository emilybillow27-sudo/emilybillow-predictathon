import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --------------------------------------------------------------
# Resolve repo root and paths
# --------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cv1_path = os.path.join(ROOT, "submission_output", "cv1_results.csv")
scatter_out = os.path.join(ROOT, "submission_output", "cv1_scatter.png")
foldwise_out = os.path.join(ROOT, "submission_output", "cv1_foldwise_accuracy.png")

# --------------------------------------------------------------
# Load CV1 results
# --------------------------------------------------------------
df = pd.read_csv(cv1_path)

# Compute Pearson correlation
r, p = pearsonr(df["value"], df["pred"])
print(f"CV1 Pearson r = {r:.3f}")

# --------------------------------------------------------------
# Scatterplot: Observed vs Predicted
# --------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="value",
    y="pred",
    hue="fold",
    palette="viridis",
    alpha=0.7,
    s=40
)

# 1:1 line
min_val = min(df["value"].min(), df["pred"].min())
max_val = max(df["value"].max(), df["pred"].max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

plt.xlabel("Observed Grain Yield")
plt.ylabel("Predicted Grain Yield")
plt.title(f"CV1 Observed vs Predicted (r = {r:.3f})")
plt.legend(title="Fold", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig(scatter_out, dpi=300)
plt.close()

# --------------------------------------------------------------
# Fold-wise accuracy bar plot
# --------------------------------------------------------------
fold_r = df.groupby("fold").apply(lambda g: pearsonr(g["value"], g["pred"])[0])

plt.figure(figsize=(6, 4))
fold_r.plot(kind="bar", color="skyblue", edgecolor="black")
plt.ylabel("Pearson r")
plt.xlabel("Fold")
plt.title("CV1 Fold-wise Accuracy")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(foldwise_out, dpi=300)
plt.close()

print("✓ CV1 plots saved.")