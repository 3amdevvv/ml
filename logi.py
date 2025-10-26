import numpy as np
import pandas as pd

# -----------------------------
# 1. Load & Understand Dataset
# -----------------------------
df = pd.read_csv("data/Iris.csv")

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Cleanup: remove ID column if unnecessary
df = df.drop(columns=['Id'])

# Encode Species as numbers
classes = df['Species'].unique()
class_to_num = {c: i for i, c in enumerate(classes)}
df['Species'] = df['Species'].map(class_to_num)

# Split into features (X) and target (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# One-hot encode labels
num_classes = len(np.unique(y))
y_onehot = np.eye(num_classes)[y]

# Add bias column
X = np.hstack((np.ones((X.shape[0], 1)), X))

# -----------------------------
# 2. Define Functions
# -----------------------------
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_cost(X, y, theta):
    m = X.shape[0]
    h = softmax(np.dot(X, theta))
    epsilon = 1e-5
    cost = -np.sum(y * np.log(h + epsilon)) / m
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = X.shape[0]
    cost_history = []
    for i in range(iterations):
        h = softmax(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    return theta, cost_history

# -----------------------------
# 3. Train Model
# -----------------------------
np.random.seed(42)
theta = np.zeros((X.shape[1], num_classes))
learning_rate = 0.1
iterations = 1000

theta, cost_history = gradient_descent(X, y_onehot, theta, learning_rate, iterations)

# -----------------------------
# 4. Evaluate Model
# -----------------------------
def predict(X, theta):
    probs = softmax(np.dot(X, theta))
    return np.argmax(probs, axis=1)

y_pred = predict(X, theta)
accuracy = np.mean(y_pred == y) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")

# -----------------------------
# 5. Plot Cost vs Iterations
# -----------------------------
import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence (Logistic Regression)")
plt.show()
