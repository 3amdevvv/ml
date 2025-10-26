# Task 35 - Decision Tree using GINI Index

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv("data/Iris.csv")  # Ensure the Iris.csv file is in the same folder
print(df.head())

# 3. Check for missing values
print(df.isnull().sum())

# 4. Identify features and target
X = df.iloc[:, 1:5]   # Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
y = df.iloc[:, -1]    # Target: Species

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Build Decision Tree Model (GINI Index)
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=2, random_state=42)
model_gini.fit(X_train, y_train)

# 7. Predict
y_pred = model_gini.predict(X_test)

# 8. Evaluate Model
print("Accuracy (GINI):", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
8888
# Task 36 - Decision Tree using ENTROPY

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Build Decision Tree Model (Entropy)
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=2, random_state=42)
model_entropy.fit(X_train, y_train)

# Predict
y_pred_entropy = model_entropy.predict(X_test)

# Evaluate Model
print("Accuracy (Entropy):", accuracy_score(y_test, y_pred_entropy))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("\nClassification Report:\n", classification_report(y_test, y_pred_entropy))
8888
# Task 37 - Decision Tree using LOG LOSS

# Build Decision Tree Model (Log Loss)
model_log = DecisionTreeClassifier(criterion='log_loss', max_depth=4, min_samples_split=2, random_state=42)
model_log.fit(X_train, y_train)

# Predict
y_pred_log = model_log.predict(X_test)

# Evaluate Model
print("Accuracy (Log Loss):", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))
8888

88
