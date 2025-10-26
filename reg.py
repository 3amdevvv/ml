import pandas as pd
import numpy as np


# 1. Load the dataset
df = pd.read_csv("data/Advertising.csv")

# Show first few rows
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Clean up: Drop rows with any NaN if found
df.dropna(inplace=True)

# Extract columns (assuming they are named 'TV', 'Radio', 'Newspaper', 'Sales')
X_tv = df['TV Ad Budget ($)'].values
X_radio = df['Radio Ad Budget ($)'].values
X_news = df['Newspaper Ad Budget ($)'].values
Y = df['Sales ($)'].values

# Function to compute RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# -----------------------------
# 1️⃣ Simple Regression (Sales vs Radio)
# -----------------------------
x = X_radio
y = Y
b1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
b0 = np.mean(y) - b1 * np.mean(x)
y_pred = b0 + b1 * x
print("\nModel 1: Sales ~ Radio")
print(f"Equation: Sales = {b0:.3f} + {b1:.3f} * Radio")
print(f"RMSE = {rmse(y, y_pred):.3f}")

# -----------------------------
# 2️⃣ Simple Regression (Sales vs TV)
# -----------------------------
x = X_tv
y = Y
b1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
b0 = np.mean(y) - b1 * np.mean(x)
y_pred = b0 + b1 * x
print("\nModel 2: Sales ~ TV")
print(f"Equation: Sales = {b0:.3f} + {b1:.3f} * TV")
print(f"RMSE = {rmse(y, y_pred):.3f}")

# -----------------------------
# 3️⃣ Simple Regression (Sales vs Newspaper)
# -----------------------------
x = X_news
y = Y
b1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
b0 = np.mean(y) - b1 * np.mean(x)
y_pred = b0 + b1 * x
print("\nModel 3: Sales ~ Newspaper")
print(f"Equation: Sales = {b0:.3f} + {b1:.3f} * Newspaper")
print(f"RMSE = {rmse(y, y_pred):.3f}")

# -----------------------------
# 4️⃣ Multiple Regression (Sales ~ Radio + TV)
# -----------------------------
X = np.column_stack((np.ones(len(X_tv)), X_radio, X_tv))
y = Y.reshape(-1, 1)
beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ beta
print("\nModel 4: Sales ~ Radio + TV")
print(f"Equation: Sales = {beta[0][0]:.3f} + {beta[1][0]:.3f}*Radio + {beta[2][0]:.3f}*TV")
print(f"RMSE = {rmse(y, y_pred):.3f}")

# -----------------------------
# 5️⃣ Multiple Regression (Sales ~ Newspaper + TV)
# -----------------------------
X = np.column_stack((np.ones(len(X_tv)), X_news, X_tv))
beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ beta
print("\nModel 5: Sales ~ Newspaper + TV")
print(f"Equation: Sales = {beta[0][0]:.3f} + {beta[1][0]:.3f}*Newspaper + {beta[2][0]:.3f}*TV")
print(f"RMSE = {rmse(y, y_pred):.3f}")

# -----------------------------
# 6️⃣ Multiple Regression (Sales ~ Newspaper + Radio)
# -----------------------------
X = np.column_stack((np.ones(len(X_tv)), X_news, X_radio))
beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ beta
print("\nModel 6: Sales ~ Newspaper + Radio")
print(f"Equation: Sales = {beta[0][0]:.3f} + {beta[1][0]:.3f}*Newspaper + {beta[2][0]:.3f}*Radio")
print(f"RMSE = {rmse(y, y_pred):.3f}")
