import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load and Understand Dataset
# -------------------------------
df=pd.read_csv("data/carr.csv")
print("Dataset Preview:\n", df.head())

# Convert categorical variables (Transmission, Owner)
df["transmission"] = df["transmission"].map({"Manual": 0, "Automatic": 1})
df["owner"] = df["owner"].map({"First Owner": 0, "Second Owner": 1})

# Select features (you can change columns for different objectives)
X = df[["year", "km_driven"]].values  # Example for year & km_driven
y = df["selling_price"].values.reshape(-1, 1)

# Normalize features for better gradient descent performance
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()

# Add bias term
X = np.c_[np.ones(X.shape[0]), X]  # add a column of ones

# -------------------------------
# 2️⃣ Define Hypothesis Function
# -------------------------------
def hypothesis(X, theta):
    return np.dot(X, theta)

# -------------------------------
# 3️⃣ Cost Function (MSE)
# -------------------------------
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# -------------------------------
# 4️⃣ Gradient Descent
# -------------------------------
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = hypothesis(X, theta)
        theta -= (alpha / m) * np.dot(X.T, (predictions - y))
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

# Initialize parameters
theta = np.zeros((X.shape[1], 1))
alpha = 0.01
iterations = 1000

# -------------------------------
# 5️⃣ Run Gradient Descent
# -------------------------------
theta_final, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("\nFinal Parameters (theta):")
print(theta_final)

# -------------------------------
# 6️⃣ Plot Convergence
# -------------------------------
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Convergence")
plt.show()

# -------------------------------
# 7️⃣ Evaluate Model
# -------------------------------
predictions = hypothesis(X, theta_final)
rmse = np.sqrt(np.mean((predictions - y)**2))
print(f"\nRMSE (Normalized Units): {rmse:.4f}")

# Plot regression (for single feature visualization, e.g., year)
plt.scatter(df["year"], y, color="blue", label="Actual")
plt.scatter(df["year"], predictions, color="red", label="Predicted")
plt.xlabel("Year")
plt.ylabel("Normalized Selling Price")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.show()
