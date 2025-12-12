"""
Gaussian Naive Bayes Model for Twitter Classification
(With extracted numerical features)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def extract_numerical_features(df):
    """Extract numerical features from tweets"""
    features = []
    
    for tweet in df['tweet_text']:
        tweet_features = {
            'length': len(tweet),
            'word_count': len(tweet.split()),
            'char_count': len(tweet),
            'exclamation_count': tweet.count('!'),
            'question_count': tweet.count('?'),
            'uppercase_count': sum(1 for c in tweet if c.isupper()),
            'digit_count': sum(1 for c in tweet if c.isdigit()),
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)),
            'mention_count': len(re.findall(r'@\w+', tweet)),
            'hashtag_count': len(re.findall(r'#\w+', tweet)),
            'avg_word_length': np.mean([len(word) for word in tweet.split()]) if tweet.split() else 0
        }
        features.append(list(tweet_features.values()))
    
    return np.array(features)

def run_gaussian_naive_bayes():
    """Run Gaussian Naive Bayes model with numerical features"""
    
    print("="*60)
    print("GAUSSIAN NAIVE BAYES MODEL (WITH NUMERICAL FEATURES)")
    print("="*60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_text(df)
    
    # Extract numerical features
    print("Extracting numerical features...")
    X = extract_numerical_features(df)
    y = df['numeric_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    feature_names = ['length', 'word_count', 'char_count', 'exclamation_count', 
                    'question_count', 'uppercase_count', 'digit_count', 'url_count',
                    'mention_count', 'hashtag_count', 'avg_word_length']
    print(f"Feature names: {feature_names}")
    
    # Initialize and train model
    print("\nTraining Gaussian Naive Bayes...")
    model = GaussianNB()
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
    
    # Print feature statistics
    print("\nFeature Statistics:")
    print("-" * 30)
    for i, feature_name in enumerate(feature_names):
        print(f"{feature_name}: Mean={X_train[:, i].mean():.2f}, Std={X_train[:, i].std():.2f}")
    
    # Create results directory
    results_dir = "Results/GaussianNB"
    
    # Save results
    save_metrics_to_file(metrics_train, metrics_test, "GaussianNB", results_dir)
    save_classification_report(y_train, y_train_pred, "GaussianNB", results_dir, "Training")
    save_classification_report(y_test, y_test_pred, "GaussianNB", results_dir, "Test")
    
    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, "GaussianNB", results_dir, "Training")
    plot_confusion_matrix(y_test, y_test_pred, "GaussianNB", results_dir, "Test")
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_train, metrics_test, "GaussianNB", results_dir)
    
    # Save feature information
    with open(f"{results_dir}/feature_info.txt", 'w') as f:
        f.write("Gaussian Naive Bayes - Numerical Features Used\n")
        f.write("=" * 50 + "\n\n")
        f.write("Features extracted from tweets:\n")
        for i, feature_name in enumerate(feature_names):
            f.write(f"{i+1}. {feature_name}: Mean={X_train[:, i].mean():.2f}, Std={X_train[:, i].std():.2f}\n")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*60)
    
    return model, metrics_train, metrics_test

if __name__ == "__main__":
    run_gaussian_naive_bayes()