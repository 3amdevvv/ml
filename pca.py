!pip install pandas numpy scikit-learn matplotlib seaborn
8888
# ============================================================
# Step 2: Load Dataset
# ============================================================
# Load your dataset (replace path if needed)
df = pd.read_csv("data/Iris.csv")

# Display first few rows
print("🔹 First 5 rows of dataset:")
display(df.head())

# Info and shape
print("\nDataset Info:")
print(df.info())

print("\nShape of dataset:", df.shape)
8888
# ============================================================
# Step 3: Preprocess Data
# ============================================================
# Drop unnecessary column if exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Features (X) and Target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Standardize features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
8888
# ============================================================
# Step 4: Classifier Without PCA (Baseline)
# ============================================================
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate performance
print("🔹 Performance WITHOUT PCA:")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Without PCA)")
plt.show()
8888
# ============================================================
# Step 5: Apply PCA for Dimensionality Reduction
# ============================================================
# Choose number of components (2D for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split transformed data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", np.sum(pca.explained_variance_ratio_))

# Visualize 2D PCA features
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1')
plt.title("PCA - 2D Feature Distribution")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
8888
# ============================================================
# Step 6: Classifier With PCA
# ============================================================
svm_pca_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_pca_model.fit(X_train_pca, y_train_pca)
y_pred_pca = svm_pca_model.predict(X_test_pca)

# Evaluate performance
print("🔹 Performance WITH PCA:")
print("Accuracy:", accuracy_score(y_test_pca, y_pred_pca) * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test_pca, y_pred_pca))
print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_test_pca, y_pred_pca), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix (With PCA)")
plt.show()
8888
# ============================================================
# Step 7: Compare Model Performance
# ============================================================
acc_without_pca = accuracy_score(y_test, y_pred)
acc_with_pca = accuracy_score(y_test_pca, y_pred_pca)

print("\n🔹 Accuracy Comparison:")
print(f"Without PCA: {acc_without_pca * 100:.2f}%")
print(f"With PCA: {acc_with_pca * 100:.2f}%")

# Bar chart for comparison
plt.bar(["Without PCA", "With PCA"], [acc_without_pca, acc_with_pca], color=['skyblue', 'lightgreen'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()


88
