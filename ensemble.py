# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Ensemble methods
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Base model
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns
8888
# Load Iris dataset
iris = load_iris()

# Convert to pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display first few rows
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Target mapping
target_names = iris.target_names
print("\nTarget Names:", target_names)
8888
# Define features (X) and target (y)
X = df.iloc[:, :-1]
y = df['target']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
8888
# exp 38 logistic
# Initialize base learner
base_model = DecisionTreeClassifier(random_state=42)

# Bagging Classifier
bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Predictions
y_pred_bag = bagging_model.predict(X_test)

# Evaluation
print("\n--- Bagging Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_bag))
print("\nClassification Report:\n", classification_report(y_test, y_pred_bag))
sns.heatmap(confusion_matrix(y_test, y_pred_bag), annot=True, cmap="Blues", fmt="d")
plt.title("Bagging Classifier Confusion Matrix")
plt.show()
8888
# exp 39
# Initialize base learner
base_model = DecisionTreeClassifier(random_state=42)

# Bagging Classifier
bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# Predictions
y_pred_bag = bagging_model.predict(X_test)

# Evaluation
print("\n--- Bagging Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_bag))
print("\nClassification Report:\n", classification_report(y_test, y_pred_bag))
sns.heatmap(confusion_matrix(y_test, y_pred_bag), annot=True, cmap="Blues", fmt="d")
plt.title("Bagging Classifier Confusion Matrix")
plt.show()
8888
#exp 40
# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("\n--- Random Forest Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="Greens", fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.show()
8888
#exp 41
# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluation
print("\n--- Gradient Boosting Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb))
sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True, cmap="Oranges", fmt="d")
plt.title("Gradient Boosting Confusion Matrix")
plt.show()
8888
#exp 42
# AdaBoost Classifier
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada_model.fit(X_train, y_train)

# Predictions
y_pred_ada = ada_model.predict(X_test)

# Evaluation
print("\n--- AdaBoost Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print("\nClassification Report:\n", classification_report(y_test, y_pred_ada))
sns.heatmap(confusion_matrix(y_test, y_pred_ada), annot=True, cmap="Purples", fmt="d")
plt.title("AdaBoost Confusion Matrix")
plt.show()

888
