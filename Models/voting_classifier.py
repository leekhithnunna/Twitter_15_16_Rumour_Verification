"""
Voting Classifier (Ensemble) Model for Twitter Classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

def run_voting_classifier():
    """Run Voting Classifier (Ensemble) model"""
    
    print("="*60)
    print("VOTING CLASSIFIER (ENSEMBLE) MODEL")
    print("="*60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_text(df)
    
    # Prepare features (TF-IDF works well with most models)
    X, y, vectorizer = prepare_features(df, feature_type='tfidf')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Initialize base classifiers
    print("\nInitializing base classifiers...")
    
    # For LDA, we need dense matrix
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    classifiers = [
        ('multinomial_nb', MultinomialNB(alpha=1.0)),
        ('logistic_reg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)),
        ('lda', LinearDiscriminantAnalysis(solver='svd'))
    ]
    
    # Create voting classifier with soft voting (uses probabilities)
    print("Training Voting Classifier (Ensemble)...")
    model = VotingClassifier(
        estimators=classifiers,
        voting='soft'  # Use probabilities for voting
    )
    
    # Fit the ensemble - we need to handle LDA separately due to dense matrix requirement
    # Create a custom fit approach
    model.estimators_ = []
    
    # Fit MultinomialNB and LogisticRegression on sparse matrix
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    model.estimators_.append(mnb)
    
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    model.estimators_.append(lr)
    
    # Fit LDA on dense matrix
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(X_train_dense, y_train)
    model.estimators_.append(lda)
    
    # Custom prediction function
    def ensemble_predict(X_sparse, X_dense):
        # Get predictions from each model
        pred1 = mnb.predict(X_sparse)
        pred2 = lr.predict(X_sparse)
        pred3 = lda.predict(X_dense)
        
        # Simple majority voting
        predictions = []
        for i in range(len(pred1)):
            votes = [pred1[i], pred2[i], pred3[i]]
            prediction = max(set(votes), key=votes.count)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    # Make predictions
    y_train_pred = ensemble_predict(X_train, X_train_dense)
    y_test_pred = ensemble_predict(X_test, X_test_dense)
    
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
    
    # Print individual classifier performance
    print("\nIndividual Classifier Performance on Test Set:")
    print("-" * 50)
    
    # MultinomialNB
    mnb_pred = mnb.predict(X_test)
    mnb_acc = accuracy_score(y_test, mnb_pred)
    print(f"Multinomial NB Accuracy: {mnb_acc:.4f}")
    
    # Logistic Regression
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    
    # LDA
    lda_pred = lda.predict(X_test_dense)
    lda_acc = accuracy_score(y_test, lda_pred)
    print(f"LDA Accuracy: {lda_acc:.4f}")
    
    print(f"Ensemble Accuracy: {metrics_test['accuracy']:.4f}")
    
    # Create results directory
    results_dir = "Results/VotingClassifier"
    
    # Save results
    save_metrics_to_file(metrics_train, metrics_test, "VotingClassifier", results_dir)
    save_classification_report(y_train, y_train_pred, "VotingClassifier", results_dir, "Training")
    save_classification_report(y_test, y_test_pred, "VotingClassifier", results_dir, "Test")
    
    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, "VotingClassifier", results_dir, "Training")
    plot_confusion_matrix(y_test, y_test_pred, "VotingClassifier", results_dir, "Test")
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_train, metrics_test, "VotingClassifier", results_dir)
    
    # Save individual classifier performance
    with open(f"{results_dir}/individual_classifier_performance.txt", 'w') as f:
        f.write("Individual Classifier Performance on Test Set\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Multinomial NB Accuracy: {mnb_acc:.4f}\n")
        f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
        f.write(f"LDA Accuracy: {lda_acc:.4f}\n")
        f.write(f"Ensemble Accuracy: {metrics_test['accuracy']:.4f}\n")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*60)
    
    return model, metrics_train, metrics_test

if __name__ == "__main__":
    run_voting_classifier()