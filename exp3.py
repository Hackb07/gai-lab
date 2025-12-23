# ==========================================
# Built-in Dataset Visualization (Iris)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# ------------------------------------------
# Load Built-in Dataset
# ------------------------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
df["target"] = iris.target

print(df.head())

# ------------------------------------------
# 1. Class Distribution
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ------------------------------------------
# 2. Feature Distribution (Histogram)
# ------------------------------------------
df.iloc[:, :-1].hist(
    bins=15,
    figsize=(10, 6),
    edgecolor="black"
)
plt.suptitle("Feature Distributions")
plt.show()

# ------------------------------------------
# 3. Scatter Plot
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=df.columns[0],
    y=df.columns[1],
    hue="target",
    data=df
)
plt.title("Feature Scatter Plot")
plt.show()

# ------------------------------------------
# 4. Correlation Heatmap
# ------------------------------------------
plt.figure(figsize=(7, 5))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()

# ------------------------------------------
# 5. Pair Plot (Complete Visualization)
# ------------------------------------------
sns.pairplot(
    df,
    hue="target",
    diag_kind="kde"
)
plt.show()

print("✅ Built-in dataset visualization completed successfully.")
