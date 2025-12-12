"""
Quadratic Discriminant Analysis Model for Twitter Classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def run_quadratic_discriminant_analysis():
    """Run Quadratic Discriminant Analysis model"""
    
    print("="*60)
    print("QUADRATIC DISCRIMINANT ANALYSIS MODEL")
    print("="*60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df = preprocess_text(df)
    
    # Prepare features (TF-IDF)
    X, y, vectorizer = prepare_features(df, feature_type='tfidf')
    
    # Convert sparse matrix to dense for QDA
    X = X.toarray()
    
    # Apply PCA for dimensionality reduction (QDA needs fewer features than samples)
    print("Applying PCA for dimensionality reduction...")
    n_components = min(100, X.shape[0] - 1, X.shape[1])  # Ensure valid number of components
    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(X)
    
    print(f"Reduced features from {X.shape[1]} to {n_components} using PCA")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution in training: {dict(zip(unique, counts))}")
    
    # Initialize and train model
    print("\nTraining Quadratic Discriminant Analysis...")
    try:
        model = QuadraticDiscriminantAnalysis(reg_param=0.01)  # Add regularization
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
        results_dir = "Results/QuadraticDiscriminantAnalysis"
        
        # Save results
        save_metrics_to_file(metrics_train, metrics_test, "QuadraticDiscriminantAnalysis", results_dir)
        save_classification_report(y_train, y_train_pred, "QuadraticDiscriminantAnalysis", results_dir, "Training")
        save_classification_report(y_test, y_test_pred, "QuadraticDiscriminantAnalysis", results_dir, "Test")
        
        # Plot confusion matrices
        plot_confusion_matrix(y_train, y_train_pred, "QuadraticDiscriminantAnalysis", results_dir, "Training")
        plot_confusion_matrix(y_test, y_test_pred, "QuadraticDiscriminantAnalysis", results_dir, "Test")
        
        # Plot metrics comparison
        plot_metrics_comparison(metrics_train, metrics_test, "QuadraticDiscriminantAnalysis", results_dir)
        
        # Save PCA information
        with open(f"{results_dir}/pca_info.txt", 'w') as f:
            f.write("Quadratic Discriminant Analysis - PCA Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original features: {vectorizer.get_feature_names_out().shape[0]}\n")
            f.write(f"PCA components: {n_components}\n")
            f.write(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}\n")
            f.write(f"Regularization parameter: 0.01\n")
        
        print(f"\nResults saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error training QDA: {e}")
        print("QDA may not be suitable for this dataset due to insufficient samples per class or numerical issues.")
        
        # Create results directory and save error info
        results_dir = "Results/QuadraticDiscriminantAnalysis"
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f"{results_dir}/error_log.txt", 'w') as f:
            f.write("Quadratic Discriminant Analysis - Error Log\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Error: {str(e)}\n")
            f.write("QDA may not be suitable for this dataset due to:\n")
            f.write("1. Insufficient samples per class\n")
            f.write("2. High dimensionality\n")
            f.write("3. Numerical instability in covariance matrix estimation\n")
        
        return None, None, None
    
    print("="*60)
    
    return model, metrics_train, metrics_test

if __name__ == "__main__":
    run_quadratic_discriminant_analysis()