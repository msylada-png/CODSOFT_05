# Credit Card Fraud Detection - Machine Learning Project

A comprehensive machine learning project to identify fraudulent credit card transactions using Logistic Regression and Random Forest classifiers.

## 📋 Project Overview

This project implements a complete fraud detection pipeline including:
- **Data Preprocessing**: Feature scaling and normalization
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Model Training**: Logistic Regression and Random Forest
- **Performance Evaluation**: Precision, Recall, F1-Score, and Accuracy metrics
- **Feature Analysis**: Feature importance ranking and prediction examples

## 📁 Project Structure

```
Credit_Card/
├── creditcard.csv                           # Dataset (must be in this folder)
├── fraud_detection.py                       # Main Python script
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)


### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis**
   ```bash
   python fraud_detection.py
   ```

Results will display in your terminal! 📊

## 📊 Project Sections

The script performs the following analysis in order:

1. **Load Dataset** - Imports and loads your creditcard.csv
2. **Explore Dataset** - Statistical analysis and class distribution
3. **Preprocess Data** - Feature scaling and normalization
4. **Handle Class Imbalance** - SMOTE balancing technique
5. **Split Dataset** - Training/testing split (80/20)
6. **Train Classification Models** - Logistic Regression & Random Forest
7. **Evaluate Model Performance** - Metrics and confusion matrices
8. **Display Example Predictions** - Real prediction examples

## 📈 Key Features

### Data Preprocessing
- StandardScaler for feature normalization
- Handles missing values
- Feature engineering ready

### Class Imbalance Solution
- SMOTE for synthetic sample generation
- Maintains test set integrity
- Balanced training data distribution

### Model Evaluation
- **Precision**: Accuracy of fraud predictions
- **Recall**: Coverage of actual fraud cases
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness
- **Confusion Matrix**: True/False Positives and Negatives

### Visualizations
- Class distribution (bar and pie charts)
- Confusion matrices heatmaps
- Metrics comparison charts
- Feature importance rankings

## 📝 Expected Output

When you run the script, you'll see:
- Dataset statistics and info
- Class distribution analysis
- Before/after SMOTE comparison
- Model training progress
- Detailed performance metrics for both models
- Confusion matrices visualization
- Feature importance analysis
- Sample prediction examples

## 💡 Model Performance


The script trains and compares:

- **Logistic Regression**: Fast, interpretable model
- **Random Forest**: More complex, typically better accuracy

Both models are evaluated on the same test set for fair comparison.

## 🔧 Customization

You can easily modify:
- `test_size`: Change train/test split ratio
- `n_estimators`: Adjust Random Forest tree count
- `max_iter`: Increase iterations for Logistic Regression
- SMOTE sampling strategy
- Feature selection parameters

## 📚 Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and metrics
- **imbalanced-learn**: SMOTE implementation
- **matplotlib & seaborn**: Data visualization


## ⚠️ Important Notes

1. **Dataset**: Ensure `creditcard.csv` is in the same folder as the script
2. **First Run**: The script may take a few minutes to run (especially SMOTE)
3. **Memory**: Large datasets may require significant RAM
4. **Random State**: Set to 42 for reproducible results
5. **Class Imbalance**: This project handles highly imbalanced fraud data

## 🎯 Results Interpretation

- **High Precision**: Few false fraud alerts (fewer innocent customers blocked)
- **High Recall**: Catches most fraud cases (better security)
- **High F1-Score**: Good balance between precision and recall
- **Confusion Matrix**: Shows True/False positives and negatives

## 🔍 Troubleshooting

**Issue**: "Module not found" error
- Solution: Run `pip install -r requirements.txt`

**Issue**: Dataset not found
- Solution: Ensure `creditcard.csv` is in the project folder

**Issue**: Out of memory error
- Solution: Reduce test size or use a subset of data

## 📊 Feature Analysis

The Random Forest model provides feature importance scores, helping identify which transaction characteristics are most important for fraud detection.

## 🚀 Next Steps for Enhancement

- Hyperparameter tuning (GridSearchCV)
- Try XGBoost or Gradient Boosting
- Implement cross-validation
- Cost-sensitive learning
- Model deployment and monitoring
- Real-time fraud detection pipeline

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Author

Created as a comprehensive machine learning project demonstration.

---

**Happy Fraud Detection! 🎉**
