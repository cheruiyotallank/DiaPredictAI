import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

print("==================================================")
print("DiaPredict AI - Diabetes Prediction ML Pipeline")
print("==================================================\n")

# 1. Load the dataset
# Column names based on Pima Indians Diabetes dataset description
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv('data/diabetes.csv', names=col_names)

print("1. DATASET OVERVIEW")
print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Outcome'].value_counts(normalize=True) * 100}\n")

# 2. Data Preprocessing & Cleaning
print("2. DATA PREPROCESSING")
# Replace mathematically impossible zeros with NaNs, then impute with median
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[features_with_zeros] = df[features_with_zeros].replace(0, np.nan)

for feature in features_with_zeros:
    median_val = df[feature].median()
    df[feature] = df[feature].fillna(median_val)

print("Handling missing/zero values: Complete.")

# Feature Selection
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Data Splitting & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaling and train/test split: Complete.\n")


# 4. Model Training & Evaluation Setup
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

results = []

print("3. MODEL TRAINING AND EVALUATION\n")
# Setup subplots for confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                xticklabels=['Non-Diabetic', 'Diabetic'], 
                yticklabels=['Non-Diabetic', 'Diabetic'])
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png')
print("Saved confusion matrices to outputs/confusion_matrices.png")

# Compare Model Performances
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
# Exclude the model column for plotting
metrics_df = results_df.set_index('Model')
metrics_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Model Performance Comparison - DiaPredict AI')
plt.ylabel('Score')
plt.ylim(0.4, 1.0)
plt.legend(loc='lower left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
print("Saved performance comparison chart to outputs/model_comparison.png")

# 5. Cross Validation for the best model (usually RF or SVM)
print("\n4. CROSS-VALIDATION (Stability Check)")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{name} 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n==================================================")
print("Pipeline Execution Completed.")
print("==================================================")
