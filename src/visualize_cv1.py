import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load CV1 results
df = pd.read_csv("submission_output/cv1_results.csv")

# Compute Pearson correlation
r, p = pearsonr(df["value"], df["pred"])
print(f"CV1 Pearson r = {r:.3f}")

# Plot
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

plt.savefig("cv1_scatter.png", dpi=300)
plt.show()

df = pd.read_csv("submission_output/cv1_results.csv")

# Compute fold-wise Pearson r
fold_r = df.groupby("fold").apply(lambda g: pearsonr(g["value"], g["pred"])[0])

# Plot
plt.figure(figsize=(6, 4))
fold_r.plot(kind="bar", color="skyblue", edgecolor="black")
plt.ylabel("Pearson r")
plt.xlabel("Fold")
plt.title("CV1 Fold-wise Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("cv1_foldwise_accuracy.png", dpi=300)
plt.show()
