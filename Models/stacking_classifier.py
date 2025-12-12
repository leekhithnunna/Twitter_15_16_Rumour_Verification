"""
Stacking Classifier Model for Twitter Classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def run_stacking_classifier():
    """Run Stacking Classifier model"""
    
    print("="*60)
    print("STACKING CLASSIFIER MODEL")
    print("="*60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_text(df)
    
    # Prepare features (TF-IDF)
    X, y, vectorizer = prepare_features(df, feature_type='tfidf')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Define base learners
    print("\nDefining base learners...")
    base_learners = [
        ('multinomial_nb', MultinomialNB(alpha=1.0)),
        ('logistic_reg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42))
    ]
    
    # Define meta-learner
    meta_learner = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    
    # Create stacking classifier
    print("Training Stacking Classifier...")
    model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,  # 5-fold cross-validation for generating meta-features
        stack_method='predict_proba',  # Use probabilities as meta-features
        n_jobs=-1,
        passthrough=False  # Don't pass original features to meta-learner
    )
    
    # Train the model
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
    
    # Evaluate base learners individually
    print("\nBase Learner Performance (5-fold CV):")
    print("-" * 40)
    
    for name, estimator in base_learners:
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate individual base learners on test set
    print("\nBase Learner Test Set Performance:")
    print("-" * 40)
    
    base_test_scores = {}
    for name, estimator in base_learners:
        estimator.fit(X_train, y_train)
        base_pred = estimator.predict(X_test)
        base_acc = accuracy_score(y_test, base_pred)
        base_test_scores[name] = base_acc
        print(f"{name}: {base_acc:.4f}")
    
    print(f"Stacking Ensemble: {metrics_test['accuracy']:.4f}")
    
    # Create results directory
    results_dir = "Results/StackingClassifier"
    
    # Save results
    save_metrics_to_file(metrics_train, metrics_test, "StackingClassifier", results_dir)
    save_classification_report(y_train, y_train_pred, "StackingClassifier", results_dir, "Training")
    save_classification_report(y_test, y_test_pred, "StackingClassifier", results_dir, "Test")
    
    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, "StackingClassifier", results_dir, "Training")
    plot_confusion_matrix(y_test, y_test_pred, "StackingClassifier", results_dir, "Test")
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_train, metrics_test, "StackingClassifier", results_dir)
    
    # Save base learner performance
    with open(f"{results_dir}/base_learner_performance.txt", 'w') as f:
        f.write("Stacking Classifier - Base Learner Performance\n")
        f.write("=" * 50 + "\n\n")
        f.write("Base Learners Used:\n")
        for name, _ in base_learners:
            f.write(f"- {name}\n")
        f.write(f"\nMeta-learner: {type(meta_learner).__name__}\n")
        f.write(f"Cross-validation folds: 5\n")
        f.write(f"Stack method: predict_proba\n\n")
        
        f.write("Test Set Performance:\n")
        f.write("-" * 30 + "\n")
        for name, score in base_test_scores.items():
            f.write(f"{name}: {score:.4f}\n")
        f.write(f"Stacking Ensemble: {metrics_test['accuracy']:.4f}\n")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*60)
    
    return model, metrics_train, metrics_test

if __name__ == "__main__":
    run_stacking_classifier()