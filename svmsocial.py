# 1. Setup Environment
# Install required libraries (run this only once)
# !pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report
)

# 3. Load Dataset
data = pd.read_csv("data/Social_Network_Ads.csv")
print("Dataset Loaded Successfully ✅")
print(data.head())

# 4. Understand and Clean Dataset
print("\nMissing Values:\n", data.isnull().sum())

# Encode Gender (Male/Female) into numeric
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male=1, Female=0

# Features (Age, EstimatedSalary, Gender) and Target (Purchased)
X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

# 5. Split Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 6. Standardize Features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================================
# 🔹 Experiments 19–21: SVM with Linear Kernel
# =====================================================================

# Initialize Linear SVM
svm_linear = SVC(kernel='linear', random_state=42)
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
accuracy_linear = accuracy_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

print("\n--- SVM (Linear Kernel) ---")
print(f"Accuracy: {accuracy_linear:.4f}")
print(f"Recall: {recall_linear:.4f}")
print(f"Precision: {precision_linear:.4f}")
print(f"F1-Score: {f1_linear:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_linear))

# Mapping:
# Exp 19 → Accuracy
# Exp 20 → Recall
# Exp 21 → Precision


# =====================================================================
# 🔹 Experiments 22–26: SVM with RBF Kernel
# =====================================================================

# Initialize RBF SVM
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
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
recall_rbf = recall_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf)

print("\n--- SVM (RBF Kernel) ---")
print(f"Accuracy: {accuracy_rbf:.4f}")
print(f"Recall: {recall_rbf:.4f}")
print(f"Precision: {precision_rbf:.4f}")
print(f"F1-Score: {f1_rbf:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rbf))

# Mapping:
# Exp 22 → F1-Measure
# Exp 23 → Accuracy
# Exp 24 → Recall
# Exp 25 → Precision
# Exp 26 → F1-Measure


# =====================================================================
# 🔹 Compare Linear vs RBF Kernel Performance
# =====================================================================

comparison = pd.DataFrame({
    'Kernel': ['Linear', 'RBF'],
    'Accuracy': [accuracy_linear, accuracy_rbf],
    'Recall': [recall_linear, recall_rbf],
    'Precision': [precision_linear, precision_rbf],
    'F1-Score': [f1_linear, f1_rbf]
})

print("\n--- SVM Performance Comparison ---")
print(comparison)

# Plot Comparison
comparison_melted = comparison.melt(id_vars='Kernel', var_name='Metric', value_name='Score')
plt.figure(figsize=(8,5))
sns.barplot(x='Metric', y='Score', hue='Kernel', data=comparison_melted, palette='coolwarm')
plt.title("SVM Linear vs RBF Kernel - Performance Comparison")
plt.ylim(0, 1.0)
plt.show()
