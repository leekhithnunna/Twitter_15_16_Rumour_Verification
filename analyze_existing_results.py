"""
Script to analyze existing model results and select the best 3 based on test accuracy
(Use this if you've already run some models and want to analyze existing results)
"""

import os
import shutil
import pandas as pd
import re
from datetime import datetime

def parse_metrics_file(file_path):
    """Parse metrics from a model's metrics.txt file"""
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract test set metrics
        test_section = content.split('TEST SET METRICS:')[1] if 'TEST SET METRICS:' in content else ""
        
        # Parse metrics using regex
        accuracy_match = re.search(r'Accuracy: ([\d.]+)', test_section)
        precision_macro_match = re.search(r'Precision \(Macro\): ([\d.]+)', test_section)
        precision_weighted_match = re.search(r'Precision \(Weighted\): ([\d.]+)', test_section)
        recall_macro_match = re.search(r'Recall \(Macro\): ([\d.]+)', test_section)
        recall_weighted_match = re.search(r'Recall \(Weighted\): ([\d.]+)', test_section)
        f1_macro_match = re.search(r'F1-Score \(Macro\): ([\d.]+)', test_section)
        f1_weighted_match = re.search(r'F1-Score \(Weighted\): ([\d.]+)', test_section)
        
        if accuracy_match:
            metrics['Test_Accuracy'] = float(accuracy_match.group(1))
        if precision_macro_match:
            metrics['Test_Precision_Macro'] = float(precision_macro_match.group(1))
        if precision_weighted_match:
            metrics['Test_Precision_Weighted'] = float(precision_weighted_match.group(1))
        if recall_macro_match:
            metrics['Test_Recall_Macro'] = float(recall_macro_match.group(1))
        if recall_weighted_match:
            metrics['Test_Recall_Weighted'] = float(recall_weighted_match.group(1))
        if f1_macro_match:
            metrics['Test_F1_Macro'] = float(f1_macro_match.group(1))
        if f1_weighted_match:
            metrics['Test_F1_Weighted'] = float(f1_weighted_match.group(1))
            
        # Also extract training metrics
        train_section = content.split('TRAINING SET METRICS:')[1].split('TEST SET METRICS:')[0] if 'TRAINING SET METRICS:' in content else ""
        
        train_accuracy_match = re.search(r'Accuracy: ([\d.]+)', train_section)
        if train_accuracy_match:
            metrics['Train_Accuracy'] = float(train_accuracy_match.group(1))
            
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return metrics

def analyze_existing_results():
    """Analyze existing results and select best 3 models"""
    
    print("="*80)
    print("ANALYZING EXISTING MODEL RESULTS")
    print("="*80)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_dir = "Results"
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found!")
        print("Please run some models first.")
        return
    
    # Find all model result directories
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    if not model_dirs:
        print(f"No model result directories found in {results_dir}")
        return
    
    print(f"Found {len(model_dirs)} model result directories:")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
    
    # Parse results from each model
    results_summary = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(results_dir, model_dir)
        
        # Look for metrics file
        metrics_files = [f for f in os.listdir(model_path) if f.endswith('_metrics.txt')]
        
        if not metrics_files:
            print(f"Warning: No metrics file found in {model_dir}")
            continue
            
        metrics_file = os.path.join(model_path, metrics_files[0])
        metrics = parse_metrics_file(metrics_file)
        
        if 'Test_Accuracy' in metrics:
            model_name = model_dir.replace('NB', ' NB').replace('Regression', ' Regression').replace('Analysis', ' Analysis').replace('Classifier', ' Classifier')
            
            results_summary.append({
                'Model_Dir': model_dir,
                'Model_Name': model_name,
                **metrics
            })
            
            print(f"✓ Parsed {model_name}: Test Accuracy = {metrics['Test_Accuracy']:.4f}")
        else:
            print(f"✗ Could not parse metrics from {model_dir}")
    
    if len(results_summary) < 3:
        print(f"\nWarning: Only {len(results_summary)} models with valid results found. Need at least 3 for comparison.")
        return
    
    # Convert to DataFrame and sort by test accuracy
    df_results = pd.DataFrame(results_summary)
    top_3_models = df_results.sort_values('Test_Accuracy', ascending=False).head(3)
    
    print("\n" + "="*80)
    print("TOP 3 MODELS BASED ON TEST ACCURACY")
    print("="*80)
    
    print(f"{'Rank':<5} {'Model':<35} {'Test Acc':<10} {'Test F1-M':<10} {'Test F1-W':<10}")
    print("-" * 75)
    
    for rank, (_, row) in enumerate(top_3_models.iterrows(), 1):
        f1_macro = row.get('Test_F1_Macro', 'N/A')
        f1_weighted = row.get('Test_F1_Weighted', 'N/A')
        f1_macro_str = f"{f1_macro:.4f}" if isinstance(f1_macro, float) else str(f1_macro)
        f1_weighted_str = f"{f1_weighted:.4f}" if isinstance(f1_weighted, float) else str(f1_weighted)
        
        print(f"{rank:<5} {row['Model_Name']:<35} {row['Test_Accuracy']:<10.4f} {f1_macro_str:<10} {f1_weighted_str:<10}")
    
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
        f.write(f"Selection Criteria: Test Accuracy\n")
        f.write(f"Total models analyzed: {len(results_summary)}\n\n")
        
        f.write("RANKING:\n")
        f.write("-" * 20 + "\n")
        
        for rank, (_, row) in enumerate(top_3_models.iterrows(), 1):
            f.write(f"\nRANK {rank}: {row['Model_Name']}\n")
            f.write(f"Test Accuracy: {row['Test_Accuracy']:.4f}\n")
            
            for metric in ['Test_F1_Macro', 'Test_F1_Weighted', 'Test_Precision_Macro', 'Test_Recall_Macro', 'Train_Accuracy']:
                if metric in row and pd.notna(row[metric]):
                    f.write(f"{metric.replace('_', ' ')}: {row[metric]:.4f}\n")
        
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
            'Test_F1_Macro': row.get('Test_F1_Macro', None),
            'Test_F1_Weighted': row.get('Test_F1_Weighted', None),
            'Test_Precision_Macro': row.get('Test_Precision_Macro', None),
            'Test_Recall_Macro': row.get('Test_Recall_Macro', None)
        })
    
    pd.DataFrame(comparison_data).to_csv(f"{best_dir}/best_three_comparison.csv", index=False)
    
    print(f"\n📊 Best 3 models results saved to: {best_dir}/")
    print(f"📄 Summary report: {summary_file}")
    print(f"📈 Detailed data: {best_dir}/best_three_detailed.csv")
    print(f"📊 Comparison data: {best_dir}/best_three_comparison.csv")
    
    # Display final summary
    print(f"\n🏆 WINNER: {top_3_models.iloc[0]['Model_Name']}")
    print(f"   Test Accuracy: {top_3_models.iloc[0]['Test_Accuracy']:.4f}")
    
    if 'Test_F1_Macro' in top_3_models.iloc[0] and pd.notna(top_3_models.iloc[0]['Test_F1_Macro']):
        print(f"   Test F1-Macro: {top_3_models.iloc[0]['Test_F1_Macro']:.4f}")
    
    print(f"🏁 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return top_3_models

if __name__ == "__main__":
    best_models = analyze_existing_results()