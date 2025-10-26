# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report
8888
# Load Iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget classes:", iris.target_names)
8888
# Features and Target
X = df.iloc[:, :-1]
y = df['target']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
8888
# Initialize SVM (Linear Kernel)
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

# Predictions
y_pred_linear = svm_linear.predict(X_test)

# Confusion Matrix
cm_linear = confusion_matrix(y_test, y_pred_linear)
sns.heatmap(cm_linear, annot=True, cmap="Blues", fmt="d")
plt.title("SVM (Linear Kernel) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluation Metrics
precision_linear = precision_score(y_test, y_pred_linear, average='macro')
recall_linear = recall_score(y_test, y_pred_linear, average='macro')
accuracy_linear = accuracy_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear, average='macro')

print("\n--- SVM (Linear Kernel) ---")
print(f"Precision: {precision_linear:.4f}")
print(f"Recall: {recall_linear:.4f}")
print(f"Accuracy: {accuracy_linear:.4f}")
print(f"F1-Score: {f1_linear:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_linear))
8888
# Initialize SVM (RBF Kernel)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)

# Predictions
y_pred_rbf = svm_rbf.predict(X_test)

# Confusion Matrix
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(cm_rbf, annot=True, cmap="Oranges", fmt="d")
plt.title("SVM (RBF Kernel) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluation Metrics
precision_rbf = precision_score(y_test, y_pred_rbf, average='macro')
recall_rbf = recall_score(y_test, y_pred_rbf, average='macro')
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf, average='macro')

print("\n--- SVM (RBF Kernel) ---")
print(f"Precision: {precision_rbf:.4f}")
print(f"Recall: {recall_rbf:.4f}")
print(f"Accuracy: {accuracy_rbf:.4f}")
print(f"F1-Score: {f1_rbf:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rbf))
8888
# Create DataFrame for comparison
comparison = pd.DataFrame({
    'Kernel': ['Linear', 'RBF'],
    'Precision': [precision_linear, precision_rbf],
    'Recall': [recall_linear, recall_rbf],
    'Accuracy': [accuracy_linear, accuracy_rbf],
    'F1-Score': [f1_linear, f1_rbf]
})

print("\n--- SVM Performance Comparison ---")
print(comparison)

# Plot comparison
comparison_melted = comparison.melt(id_vars='Kernel', var_name='Metric', value_name='Score')
plt.figure(figsize=(8,5))
sns.barplot(x='Metric', y='Score', hue='Kernel', data=comparison_melted, palette='coolwarm')
plt.title("SVM Kernel Comparison on Iris Dataset")
plt.ylim(0.9, 1.0)
plt.show()


888
