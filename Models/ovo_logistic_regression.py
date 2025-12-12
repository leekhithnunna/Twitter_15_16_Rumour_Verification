"""
One-vs-One Logistic Regression Model for Twitter Classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_ovo_logistic_regression():
    """Run One-vs-One Logistic Regression model"""
    
    print("="*60)
    print("ONE-VS-ONE LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_text(df)
    
    # Prepare features (TF-IDF works well with Logistic Regression)
    X, y, vectorizer = prepare_features(df, feature_type='tfidf')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Initialize and train model
    print("\nTraining One-vs-One Logistic Regression...")
    base_classifier = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    model = OneVsOneClassifier(base_classifier)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred)
    
    # Print metrics
    print_metrics(metrics_train, "Training")
    print_metrics(metrics_test, "Test")
    
    # Print classification reports
    print("\nTraining Set Classification Report:")
    target_names = ['False', 'True', 'Unverified', 'Non-rumor']
    print(classification_report(y_train, y_train_pred, target_names=target_names))
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Create results directory
    results_dir = "Results/OvOLogisticRegression"
    
    # Save results
    save_metrics_to_file(metrics_train, metrics_test, "OvOLogisticRegression", results_dir)
    save_classification_report(y_train, y_train_pred, "OvOLogisticRegression", results_dir, "Training")
    save_classification_report(y_test, y_test_pred, "OvOLogisticRegression", results_dir, "Test")
    
    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, "OvOLogisticRegression", results_dir, "Training")
    plot_confusion_matrix(y_test, y_test_pred, "OvOLogisticRegression", results_dir, "Test")
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_train, metrics_test, "OvOLogisticRegression", results_dir)
    
    print(f"\nResults saved to: {results_dir}")
    print("="*60)
    
    return model, metrics_train, metrics_test

if __name__ == "__main__":
    run_ovo_logistic_regression()