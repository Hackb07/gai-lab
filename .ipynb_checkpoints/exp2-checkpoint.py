# =====================================================
# Dataset Preprocessing and Cleaning using Iris Dataset
# =====================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------------------
# 1. Load Built-in Dataset
# -----------------------------------------------------
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

print("Initial Dataset Shape:", df.shape)
print("\nDataset Preview:\n", df.head())

# -----------------------------------------------------
# 2. Data Cleaning
# -----------------------------------------------------

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Introduce artificial noise (for demo)
df.iloc[10:15, 0] = np.nan

# Handle missing values (Mean Imputation)
df.fillna(df.mean(), inplace=True)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# -----------------------------------------------------
# 3. Feature Scaling
# -----------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------
# 4. Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# -----------------------------------------------------
# 5. Visualization (Cleaned Data)
# -----------------------------------------------------
sns.pairplot(
    df,
    hue="target",
    diag_kind="kde"
)
plt.show()

# -----------------------------------------------------
# 6. Final Output Summary
# -----------------------------------------------------
print("\n✅ Dataset successfully cleaned and preprocessed.")
