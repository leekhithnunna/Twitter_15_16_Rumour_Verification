import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the Twitter dataset"""
    # Try different paths depending on where the script is run from
    possible_paths = [
        '../combined_twitter_dataset.xlsx',  # When run from Models directory
        'combined_twitter_dataset.xlsx',     # When run from root directory
        './combined_twitter_dataset.xlsx'    # Alternative root directory path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            return df
    
    # If none of the paths work, raise an error with helpful message
    raise FileNotFoundError(
        "Could not find 'combined_twitter_dataset.xlsx'. "
        "Please ensure the file exists in the root directory."
    )

def preprocess_text(df):
    """Basic text preprocessing"""
    # Simple text cleaning
    df['tweet_text'] = df['tweet_text'].str.lower()
    df['tweet_text'] = df['tweet_text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
    df['tweet_text'] = df['tweet_text'].str.replace(r'@\w+', '', regex=True)     # Remove mentions
    df['tweet_text'] = df['tweet_text'].str.replace(r'#\w+', '', regex=True)     # Remove hashtags
    df['tweet_text'] = df['tweet_text'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
    return df

def prepare_features(df, feature_type='tfidf'):
    """Prepare features for training"""
    X = df['tweet_text']
    y = df['numeric_label']
    
    if feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    elif feature_type == 'count':
        vectorizer = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    elif feature_type == 'binary':
        vectorizer = CountVectorizer(max_features=5000, stop_words='english', binary=True)
    
    X_vectorized = vectorizer.fit_transform(X)
    
    return X_vectorized, y, vectorizer

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def calculate_metrics(y_true, y_pred):
    """Calculate all required metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def print_metrics(metrics, dataset_type=""):
    """Print metrics in a formatted way"""
    print(f"\n{dataset_type} Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")

def save_metrics_to_file(metrics_train, metrics_test, model_name, results_dir):
    """Save metrics to a text file"""
    # Ensure we're using the correct path regardless of where script is run from
    if not os.path.isabs(results_dir):
        if results_dir.startswith('../'):
            # If running from Models directory, adjust path
            results_dir = results_dir.replace('../', '')
        # Ensure Results directory is in the root
        if not results_dir.startswith('Results'):
            results_dir = os.path.join('Results', results_dir.split('/')[-1])
    
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/{model_name}_metrics.txt", 'w') as f:
        f.write(f"{model_name} - Performance Metrics\n")
        f.write("="*50 + "\n\n")
        
        f.write("TRAINING SET METRICS:\n")
        f.write(f"Accuracy: {metrics_train['accuracy']:.4f}\n")
        f.write(f"Precision (Macro): {metrics_train['precision_macro']:.4f}\n")
        f.write(f"Precision (Weighted): {metrics_train['precision_weighted']:.4f}\n")
        f.write(f"Recall (Macro): {metrics_train['recall_macro']:.4f}\n")
        f.write(f"Recall (Weighted): {metrics_train['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Macro): {metrics_train['f1_macro']:.4f}\n")
        f.write(f"F1-Score (Weighted): {metrics_train['f1_weighted']:.4f}\n\n")
        
        f.write("TEST SET METRICS:\n")
        f.write(f"Accuracy: {metrics_test['accuracy']:.4f}\n")
        f.write(f"Precision (Macro): {metrics_test['precision_macro']:.4f}\n")
        f.write(f"Precision (Weighted): {metrics_test['precision_weighted']:.4f}\n")
        f.write(f"Recall (Macro): {metrics_test['recall_macro']:.4f}\n")
        f.write(f"Recall (Weighted): {metrics_test['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Macro): {metrics_test['f1_macro']:.4f}\n")
        f.write(f"F1-Score (Weighted): {metrics_test['f1_weighted']:.4f}\n")

def plot_confusion_matrix(y_true, y_pred, model_name, results_dir, dataset_type="Test"):
    """Plot and save confusion matrix"""
    # Fix results directory path
    if not os.path.isabs(results_dir):
        if results_dir.startswith('../'):
            results_dir = results_dir.replace('../', '')
        if not results_dir.startswith('Results'):
            results_dir = os.path.join('Results', results_dir.split('/')[-1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False', 'True', 'Unverified', 'Non-rumor'],
                yticklabels=['False', 'True', 'Unverified', 'Non-rumor'])
    plt.title(f'{model_name} - {dataset_type} Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/{model_name}_{dataset_type.lower()}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_report(y_true, y_pred, model_name, results_dir, dataset_type="Test"):
    """Save classification report to file"""
    # Fix results directory path
    if not os.path.isabs(results_dir):
        if results_dir.startswith('../'):
            results_dir = results_dir.replace('../', '')
        if not results_dir.startswith('Results'):
            results_dir = os.path.join('Results', results_dir.split('/')[-1])
    
    target_names = ['False', 'True', 'Unverified', 'Non-rumor']
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/{model_name}_{dataset_type.lower()}_classification_report.txt", 'w') as f:
        f.write(f"{model_name} - {dataset_type} Set Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)

def plot_metrics_comparison(metrics_train, metrics_test, model_name, results_dir):
    """Plot training vs test metrics comparison"""
    # Fix results directory path
    if not os.path.isabs(results_dir):
        if results_dir.startswith('../'):
            results_dir = results_dir.replace('../', '')
        if not results_dir.startswith('Results'):
            results_dir = os.path.join('Results', results_dir.split('/')[-1])
    
    metrics_names = ['Accuracy', 'Precision (Macro)', 'Precision (Weighted)', 
                    'Recall (Macro)', 'Recall (Weighted)', 'F1 (Macro)', 'F1 (Weighted)']
    
    train_values = [metrics_train['accuracy'], metrics_train['precision_macro'], 
                   metrics_train['precision_weighted'], metrics_train['recall_macro'],
                   metrics_train['recall_weighted'], metrics_train['f1_macro'], 
                   metrics_train['f1_weighted']]
    
    test_values = [metrics_test['accuracy'], metrics_test['precision_macro'], 
                  metrics_test['precision_weighted'], metrics_test['recall_macro'],
                  metrics_test['recall_weighted'], metrics_test['f1_macro'], 
                  metrics_test['f1_weighted']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} - Training vs Test Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/{model_name}_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()