"""
Credit Card Fraud Detection - Standalone Python Script
This script performs fraud detection without requiring Jupyter Notebook
Run with: python fraud_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREDIT CARD FRAUD DETECTION - STANDALONE SCRIPT")
print("="*80)

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================
print("\n[1/8] Loading dataset...")
try:
    df = pd.read_csv('creditcard.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.shape[1]}")
except FileNotFoundError:
    print("✗ Error: creditcard.csv not found!")
    print("  Please ensure creditcard.csv is in the same directory as this script")
    exit()

# ============================================================================
# SECTION 2: EXPLORE DATA
# ============================================================================
print("\n[2/8] Exploring dataset...")
print(f"\nDataset Info:")
print(f"  Total samples: {len(df)}")
print(f"  Total features: {df.shape[1]}")
print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nClass Distribution:")
class_dist = df['Class'].value_counts()
for cls, count in class_dist.items():
    percentage = (count / len(df)) * 100
    label = "Fraudulent" if cls == 1 else "Genuine"
    print(f"  {label}: {count} ({percentage:.2f}%)")

# ============================================================================
# SECTION 3: PREPROCESS DATA
# ============================================================================
print("\n[3/8] Preprocessing data...")
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(f"✓ Feature scaling applied")
print(f"  Features standardized using StandardScaler")

# ============================================================================
# SECTION 4: HANDLE CLASS IMBALANCE
# ============================================================================
print("\n[4/8] Splitting and balancing dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Initial split completed")
print(f"  Training set: {len(X_train)} samples")
print(f"  Testing set: {len(X_test)} samples")
print(f"  Training class distribution: {y_train.value_counts().to_dict()}")

print(f"\n✓ Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"  After SMOTE: {len(X_train_balanced)} training samples")
print(f"  Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# ============================================================================
# SECTION 5: TRAIN MODELS
# ============================================================================
print("\n[5/8] Training classification models...")

print(f"  Training Logistic Regression...")
log_reg = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_balanced, y_train_balanced)
print(f"  ✓ Logistic Regression trained")

print(f"  Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_balanced, y_train_balanced)
print(f"  ✓ Random Forest trained")

# ============================================================================
# SECTION 6: MAKE PREDICTIONS
# ============================================================================
print("\n[6/8] Making predictions...")
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)
print(f"✓ Predictions completed")

# ============================================================================
# SECTION 7: EVALUATE MODELS
# ============================================================================
print("\n[7/8] Evaluating model performance...")

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{'-'*70}")
    print(f"{model_name}")
    print(f"{'-'*70}")
    print(f"{'Metric':<20} {'Score':<20} {'Details'}")
    print(f"{'-'*70}")
    print(f"{'Accuracy':<20} {accuracy:<20.4f} Overall correctness")
    print(f"{'Precision':<20} {precision:<20.4f} Fraud prediction accuracy")
    print(f"{'Recall':<20} {recall:<20.4f} Fraud detection rate")
    print(f"{'F1-Score':<20} {f1:<20.4f} Balance of precision & recall")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'Confusion Matrix:'}")
    print(f"{'':15} {'Predicted Genuine':<20} {'Predicted Fraud':<20}")
    print(f"{'Actual Genuine':<15} {cm[0,0]:<20} {cm[0,1]:<20}")
    print(f"{'Actual Fraud':<15} {cm[1,0]:<20} {cm[1,1]:<20}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'Classification Details:'}")
    print(f"  True Negatives:  {tn} (correctly classified as genuine)")
    print(f"  False Positives: {fp} (incorrectly classified as fraud)")
    print(f"  False Negatives: {fn} (undetected fraud cases)")
    print(f"  True Positives:  {tp} (correctly detected fraud)")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

log_reg_metrics = evaluate_model(y_test, y_pred_log_reg, "LOGISTIC REGRESSION")
rf_metrics = evaluate_model(y_test, y_pred_rf, "RANDOM FOREST CLASSIFIER")

# ============================================================================
# SECTION 8: COMPARISON AND SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

comparison = pd.DataFrame({
    'Logistic Regression': log_reg_metrics,
    'Random Forest': rf_metrics
})

print(f"\n{'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
print(f"{'-'*70}")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    print(f"{metric.capitalize():<20} {log_reg_metrics[metric]:<25.4f} {rf_metrics[metric]:<25.4f}")

best_model = "Random Forest" if rf_metrics['f1'] > log_reg_metrics['f1'] else "Logistic Regression"
print(f"\n✓ Best performing model: {best_model}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("TOP 15 IMPORTANT FEATURES (Random Forest)")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n{'Feature':<20} {'Importance':<20}")
print(f"{'-'*40}")
for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['Feature']:<20} {row['Importance']:<20.6f}")

# ============================================================================
# SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (First 10 Test Cases)")
print("="*70)

label_map = {0: 'Genuine', 1: 'Fraudulent'}
print(f"\n{'Index':<10} {'Actual':<15} {'Log. Regression':<20} {'Random Forest':<20}")
print(f"{'-'*65}")

for i in range(min(10, len(y_test))):
    actual = label_map[y_test.values[i]]
    log_pred = label_map[y_pred_log_reg[i]]
    rf_pred = label_map[y_pred_rf[i]]
    print(f"{i:<10} {actual:<15} {log_pred:<20} {rf_pred:<20}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✓ FRAUD DETECTION PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"""
Summary:
  • Dataset: {len(df)} transactions analyzed
  • Features: {df.shape[1]} transaction attributes
  • Train/Test Split: 80/20 with stratification
  • Class Balance: SMOTE applied to training data
  • Models Trained: Logistic Regression + Random Forest
  • Evaluation Metrics: Precision, Recall, F1-Score, Accuracy

Results:
  • Logistic Regression F1-Score: {log_reg_metrics['f1']:.4f}
  • Random Forest F1-Score: {rf_metrics['f1']:.4f}
  • Best Model: {best_model}

Next Steps:
  1. Try hyperparameter tuning (GridSearchCV)
  2. Experiment with XGBoost or Gradient Boosting
  3. Implement cross-validation for robust evaluation
  4. Deploy model for real-time fraud detection
  5. Monitor model performance in production

For more details, check README.md or run the Jupyter notebook!
""")

print("="*80)
print("End of script. Thank you for using Fraud Detection Pipeline!")
print("="*80)
