"""
Script to run all models, select the top 3 based on test accuracy,
and save their results to a "Best_of_three" folder
"""

import os
import sys
import time
import shutil
import pandas as pd
from datetime import datetime

# Add Models directory to path
sys.path.append('Models')

# Import all model functions
from multinomial_naive_bayes import run_multinomial_naive_bayes
from bernoulli_naive_bayes import run_bernoulli_naive_bayes
from multinomial_logistic_regression import run_multinomial_logistic_regression
from ovr_logistic_regression import run_ovr_logistic_regression
from ovo_logistic_regression import run_ovo_logistic_regression
from linear_discriminant_analysis import run_linear_discriminant_analysis
from softmax_regression import run_softmax_regression
from voting_classifier import run_voting_classifier
from gaussian_naive_bayes import run_gaussian_naive_bayes
from quadratic_discriminant_analysis import run_quadratic_discriminant_analysis
from stacking_classifier import run_stacking_classifier

def run_all_and_select_best():
    """Run all models and select the best 3 based on test accuracy"""
    
    print("="*80)
    print("RUNNING ALL MODELS TO SELECT BEST 3 PERFORMERS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define all models to run
    models_to_run = [
        ("MultinomialNB", "Multinomial Naive Bayes", run_multinomial_naive_bayes),
        ("BernoulliNB", "Bernoulli Naive Bayes", run_bernoulli_naive_bayes),
        ("MultinomialLogisticRegression", "Multinomial Logistic Regression", run_multinomial_logistic_regression),
        ("OvRLogisticRegression", "One-vs-Rest Logistic Regression", run_ovr_logistic_regression),
        ("OvOLogisticRegression", "One-vs-One Logistic Regression", run_ovo_logistic_regression),
        ("LinearDiscriminantAnalysis", "Linear Discriminant Analysis", run_linear_discriminant_analysis),
        ("SoftmaxRegression", "Softmax Regression", run_softmax_regression),
        ("VotingClassifier", "Voting Classifier", run_voting_classifier),
        ("GaussianNB", "Gaussian Naive Bayes", run_gaussian_naive_bayes),
        ("QuadraticDiscriminantAnalysis", "Quadratic Discriminant Analysis", run_quadratic_discriminant_analysis),
        ("StackingClassifier", "Stacking Classifier", run_stacking_classifier)
    ]
    
    # Store results for comparison
    results_summary = []
    
    # Run each model
    for i, (model_dir, model_name, model_function) in enumerate(models_to_run, 1):
        print(f"\n[{i}/{len(models_to_run)}] Running {model_name}...")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            model, metrics_train, metrics_test = model_function()
            end_time = time.time()
            execution_time = end_time - start_time
            
            if metrics_test is not None:  # Some models might fail
                results_summary.append({
                    'Model_Dir': model_dir,
                    'Model_Name': model_name,
                    'Train_Accuracy': metrics_train['accuracy'],
                    'Test_Accuracy': metrics_test['accuracy'],
                    'Train_F1_Macro': metrics_train['f1_macro'],
                    'Test_F1_Macro': metrics_test['f1_macro'],
                    'Train_F1_Weighted': metrics_train['f1_weighted'],
                    'Test_F1_Weighted': metrics_test['f1_weighted'],
                    'Train_Precision_Macro': metrics_train['precision_macro'],
                    'Test_Precision_Macro': metrics_test['precision_macro'],
                    'Train_Recall_Macro': metrics_train['recall_macro'],
                    'Test_Recall_Macro': metrics_test['recall_macro'],
                    'Execution_Time': execution_time,
                    'Status': 'Success'
                })
                
                print(f"✓ {model_name} completed successfully")
                print(f"  Test Accuracy: {metrics_test['accuracy']:.4f}")
                print(f"  Execution Time: {execution_time:.2f} seconds")
            else:
                print(f"✗ {model_name} failed")
                
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"✗ {model_name} failed with error: {str(e)}")
    
    # Convert to DataFrame and filter successful models
    df_results = pd.DataFrame(results_summary)
    successful_models = df_results[df_results['Status'] == 'Success'].copy()
    
    if len(successful_models) < 3:
        print(f"\nWarning: Only {len(successful_models)} models succeeded. Need at least 3 for comparison.")
        return
    
    # Sort by test accuracy and select top 3
    top_3_models = successful_models.sort_values('Test_Accuracy', ascending=False).head(3)
    
    print("\n" + "="*80)
    print("TOP 3 MODELS BASED ON TEST ACCURACY")
    print("="*80)
    
    print(f"{'Rank':<5} {'Model':<35} {'Test Acc':<10} {'Test F1-M':<10} {'Test F1-W':<10}")
    print("-" * 75)
    
    for rank, (_, row) in enumerate(top_3_models.iterrows(), 1):
        print(f"{rank:<5} {row['Model_Name']:<35} {row['Test_Accuracy']:<10.4f} {row['Test_F1_Macro']:<10.4f} {row['Test_F1_Weighted']:<10.4f}")
    
    # Create Best_of_three directory
    best_dir = "Best_of_three"
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir)
    
    print(f"\nCopying results to {best_dir} folder...")
    
    # Copy results for top 3 models
    for rank, (_, row) in enumerate(top_3_models.iterrows(), 1):
        model_dir = row['Model_Dir']
        model_name = row['Model_Name']
        
        source_dir = f"Results/{model_dir}"
        dest_dir = f"{best_dir}/Rank_{rank}_{model_dir}"
        
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, dest_dir)
            print(f"✓ Rank {rank}: {model_name} -> {dest_dir}")
        else:
            print(f"✗ Warning: {source_dir} not found")
    
    # Create summary report for best 3 models
    summary_file = f"{best_dir}/best_three_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TOP 3 MODELS PERFORMANCE SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Selection Criteria: Test Accuracy\n\n")
        
        f.write("RANKING:\n")
        f.write("-" * 20 + "\n")
        
        for rank, (_, row) in enumerate(top_3_models.iterrows(), 1):
            f.write(f"\nRANK {rank}: {row['Model_Name']}\n")
            f.write(f"Test Accuracy: {row['Test_Accuracy']:.4f}\n")
            f.write(f"Test F1-Macro: {row['Test_F1_Macro']:.4f}\n")
            f.write(f"Test F1-Weighted: {row['Test_F1_Weighted']:.4f}\n")
            f.write(f"Test Precision-Macro: {row['Test_Precision_Macro']:.4f}\n")
            f.write(f"Test Recall-Macro: {row['Test_Recall_Macro']:.4f}\n")
            f.write(f"Training Accuracy: {row['Train_Accuracy']:.4f}\n")
            f.write(f"Execution Time: {row['Execution_Time']:.2f} seconds\n")
        
        f.write(f"\nBEST PERFORMING MODEL: {top_3_models.iloc[0]['Model_Name']}\n")
        f.write(f"Best Test Accuracy: {top_3_models.iloc[0]['Test_Accuracy']:.4f}\n")
    
    # Save detailed CSV
    top_3_models.to_csv(f"{best_dir}/best_three_detailed.csv", index=False)
    
    # Create comparison chart data
    comparison_data = []
    for rank, (_, row) in enumerate(top_3_models.iterrows(), 1):
        comparison_data.append({
            'Rank': rank,
            'Model': row['Model_Name'],
            'Test_Accuracy': row['Test_Accuracy'],
            'Test_F1_Macro': row['Test_F1_Macro'],
            'Test_F1_Weighted': row['Test_F1_Weighted'],
            'Test_Precision_Macro': row['Test_Precision_Macro'],
            'Test_Recall_Macro': row['Test_Recall_Macro']
        })
    
    pd.DataFrame(comparison_data).to_csv(f"{best_dir}/best_three_comparison.csv", index=False)
    
    print(f"\n📊 Best 3 models results saved to: {best_dir}/")
    print(f"📄 Summary report: {summary_file}")
    print(f"📈 Detailed data: {best_dir}/best_three_detailed.csv")
    print(f"📊 Comparison data: {best_dir}/best_three_comparison.csv")
    
    # Display final summary
    print(f"\n🏆 WINNER: {top_3_models.iloc[0]['Model_Name']}")
    print(f"   Test Accuracy: {top_3_models.iloc[0]['Test_Accuracy']:.4f}")
    print(f"   Test F1-Macro: {top_3_models.iloc[0]['Test_F1_Macro']:.4f}")
    
    total_time = sum([row['Execution_Time'] for _, row in top_3_models.iterrows()])
    print(f"\n⏱️  Total execution time for all models: {sum([r['Execution_Time'] for r in results_summary]):.2f} seconds")
    print(f"⏱️  Top 3 models execution time: {total_time:.2f} seconds")
    print(f"🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return top_3_models

if __name__ == "__main__":
    best_models = run_all_and_select_best()