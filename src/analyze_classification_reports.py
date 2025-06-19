import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_classification_report(file_path):
    """Parse a classification report file and extract metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract metrics using regex
    lines = content.strip().split('\n')
    
    # Find the class-specific lines (skip header and summary lines)
    class_lines = []
    for line in lines:
        if line.strip() and not line.startswith('---') and not line.startswith('accuracy') and not line.startswith('macro avg') and not line.startswith('weighted avg') and not line.startswith('Accuracy:') and not line.startswith('Prediction'):
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] != 'precision':
                class_lines.append(parts)
    
    results = {}
    for line in class_lines:
        if len(line) >= 5:
            # Join all parts except the last 4 numeric columns to get the full class name
            class_name = ' '.join(line[:-4])
            precision = float(line[-4])
            recall = float(line[-3])
            f1 = float(line[-2])
            support = int(line[-1])
            
            # Store metrics for both classes
            results[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            }
    
    # Extract overall accuracy
    accuracy_match = re.search(r'Accuracy: ([\d.]+)', content)
    if accuracy_match:
        results['overall_accuracy'] = float(accuracy_match.group(1))
    
    # Extract probability statistics
    prob_stats = {}
    prob_lines = content.split('Prediction probabilities summary:')[-1].strip().split('\n')
    for line in prob_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = float(value.strip())
            prob_stats[key] = value
    
    results['probability_stats'] = prob_stats
    
    return results

def analyze_all_reports():
    """Analyze all classification reports in the reports directory."""
    all_results = []
    
    # Walk through the reports directory
    for root, _, files in os.walk('reports'):
        for file in files:
            if file != 'classification_report.txt':
                continue
            
            file_path = os.path.join(root, file)
            
            # Extract segment and mode from path
            path_parts = file_path.split(os.sep)
            
            # Skip if not an XGBoost report with max_depth_10
            if 'xg_boost' not in path_parts or 'max_depth_10' not in path_parts:
                continue
            
            # Extract segment and mode
            segment = None
            mode = None
            depth = None
            
            for part in path_parts:
                if part.startswith('segment_'):
                    segment = part.replace('segment_', '')
                elif part.startswith('mode_'):
                    mode = part.replace('mode_', '')
                elif part.startswith('max_depth_'):
                    depth = part.replace('max_depth_', '')
            
            if not all([segment, mode, depth]):
                continue
            
            try:
                results = parse_classification_report(file_path)
                
                # Find the positive class (the one with "not preselected")
                positive_class_name = None
                for class_name in results.keys():
                    if class_name != 'overall_accuracy' and class_name != 'probability_stats':
                        if '(not preselected)' in class_name:
                            positive_class_name = class_name
                            break
                
                if positive_class_name and positive_class_name in results:
                    positive_metrics = results[positive_class_name]
                    
                    all_results.append({
                        'segment': segment,
                        'mode': mode,
                        'max_depth': int(depth),
                        'positive_class': positive_class_name,
                        'precision': positive_metrics['precision'],
                        'recall': positive_metrics['recall'],
                        'f1_score': positive_metrics['f1_score'],
                        'support': positive_metrics['support'],
                        'accuracy': results.get('overall_accuracy', np.nan),
                        'mean_probability': results.get('probability_stats', {}).get('mean_probability', np.nan),
                        'std_probability': results.get('probability_stats', {}).get('std_probability', np.nan)
                    })
                    
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
    
    return pd.DataFrame(all_results)

def create_analysis_plots(df):
    """Create visualization plots for the analysis focusing on positive class metrics."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("RdYlGn")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('XGBoost Model Performance Analysis - Positive Class Metrics', fontsize=16, fontweight='bold')
    
    # 1. Precision heatmap
    ax1 = axes[0, 0]
    pivot_precision = df.pivot_table(index='mode', columns='segment', values='precision', aggfunc='mean')
    sns.heatmap(pivot_precision, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, center=0.5)
    ax1.set_title('Positive Class Precision by Mode and Segment')
    
    # 2. Recall heatmap
    ax2 = axes[0, 1]
    pivot_recall = df.pivot_table(index='mode', columns='segment', values='recall', aggfunc='mean')
    sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2, center=0.5)
    ax2.set_title('Positive Class Recall by Mode and Segment')
    
    # 3. F1 Score heatmap
    ax3 = axes[0, 2]
    pivot_f1 = df.pivot_table(index='mode', columns='segment', values='f1_score', aggfunc='mean')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3, center=0.5)
    ax3.set_title('Positive Class F1 Score by Mode and Segment')
    
    # 4. Support (class distribution) heatmap
    ax4 = axes[1, 0]
    support_by_mode = df.pivot_table(index='mode', columns='segment', values='support', aggfunc='mean')
    sns.heatmap(support_by_mode, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Positive Class Support by Mode and Segment')
    
    # 5. Mean Probability heatmap
    ax5 = axes[1, 1]
    prob_pivot = df.pivot_table(index='mode', columns='segment', values='mean_probability', aggfunc='mean')
    sns.heatmap(prob_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax5, center=0.5)
    ax5.set_title('Mean Prediction Probability by Mode and Segment')
    
    # 6. Standard Deviation of Probability heatmap
    ax6 = axes[1, 2]
    std_pivot = df.pivot_table(index='mode', columns='segment', values='std_probability', aggfunc='mean')
    sns.heatmap(std_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6)
    ax6.set_title('Std Dev of Prediction Probability by Mode and Segment')
    
    plt.tight_layout()
    plt.savefig('plots/positive_class_analysis_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('plots/positive_class_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_analysis(df):
    """Print detailed analysis of the results focusing on positive class metrics."""
    print("=" * 80)
    print("XGBOOST CLASSIFICATION MODEL ANALYSIS - POSITIVE CLASS METRICS")
    print("=" * 80)
    
    print(f"\nTotal models analyzed: {len(df)}")
    print(f"Segments: {df['segment'].unique()}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Max depths: {df['max_depth'].unique()}")
    
    print("\n" + "=" * 50)
    print("POSITIVE CLASS PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Overall averages for positive class
    print(f"\nPositive Class Average Metrics:")
    print(f"  Precision: {df['precision'].mean():.3f}")
    print(f"  Recall: {df['recall'].mean():.3f}")
    print(f"  F1 Score: {df['f1_score'].mean():.3f}")
    
    print("\n" + "=" * 50)
    print("POSITIVE CLASS PERFORMANCE BY SEGMENT")
    print("=" * 50)
    
    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]
        print(f"\n{segment.upper()} SEGMENT:")
        print(f"  Number of models: {len(segment_data)}")
        print(f"  Average Precision: {segment_data['precision'].mean():.3f}")
        print(f"  Average Recall: {segment_data['recall'].mean():.3f}")
        print(f"  Average F1 Score: {segment_data['f1_score'].mean():.3f}")
        
        # Best performing mode in this segment
        best_f1_idx = segment_data['f1_score'].idxmax()
        best_mode = segment_data.loc[best_f1_idx, 'mode']
        best_f1 = segment_data.loc[best_f1_idx, 'f1_score']
        best_precision = segment_data.loc[best_f1_idx, 'precision']
        best_recall = segment_data.loc[best_f1_idx, 'recall']
        print(f"  Best performing mode: {best_mode}")
        print(f"    - F1: {best_f1:.3f}")
        print(f"    - Precision: {best_precision:.3f}")
        print(f"    - Recall: {best_recall:.3f}")
    
    print("\n" + "=" * 50)
    print("POSITIVE CLASS PERFORMANCE BY MODE")
    print("=" * 50)
    
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode]
        print(f"\n{mode.upper()} MODE:")
        print(f"  Number of models: {len(mode_data)}")
        print(f"  Average Precision: {mode_data['precision'].mean():.3f}")
        print(f"  Average Recall: {mode_data['recall'].mean():.3f}")
        print(f"  Average F1 Score: {mode_data['f1_score'].mean():.3f}")
        print(f"  Average Support: {mode_data['support'].mean():.0f}")
        
        # Best performing segment for this mode
        if len(mode_data) > 1:
            best_f1_idx = mode_data['f1_score'].idxmax()
            best_segment = mode_data.loc[best_f1_idx, 'segment']
            best_f1 = mode_data.loc[best_f1_idx, 'f1_score']
            best_precision = mode_data.loc[best_f1_idx, 'precision']
            best_recall = mode_data.loc[best_f1_idx, 'recall']
            print(f"  Best performing segment: {best_segment}")
            print(f"    - F1: {best_f1:.3f}")
            print(f"    - Precision: {best_precision:.3f}")
            print(f"    - Recall: {best_recall:.3f}")
    
    print("\n" + "=" * 50)
    print("DETAILED MODEL PERFORMANCE")
    print("=" * 50)
    
    # Sort by F1 score and show top performers
    df_sorted = df.sort_values('f1_score', ascending=False)
    print(f"\n🏆 TOP 5 PERFORMERS (by F1 Score):")
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"  {i}. {row['segment']}/{row['mode']} (max_depth={row['max_depth']})")
        print(f"     - F1: {row['f1_score']:.3f}")
        print(f"     - Precision: {row['precision']:.3f}")
        print(f"     - Recall: {row['recall']:.3f}")
        print(f"     - Support: {row['support']:.0f}")
    
    print(f"\n❌ BOTTOM 5 PERFORMERS (by F1 Score):")
    for i, (_, row) in enumerate(df_sorted.tail(5).iterrows(), 1):
        print(f"  {i}. {row['segment']}/{row['mode']} (max_depth={row['max_depth']})")
        print(f"     - F1: {row['f1_score']:.3f}")
        print(f"     - Precision: {row['precision']:.3f}")
        print(f"     - Recall: {row['recall']:.3f}")
        print(f"     - Support: {row['support']:.0f}")
    
    print("\n" + "=" * 50)
    print("PRECISION-FOCUSED ANALYSIS")
    print("=" * 50)
    
    # Sort by precision and show top performers
    df_precision_sorted = df.sort_values('precision', ascending=False)
    print(f"\n🎯 TOP 5 PRECISION PERFORMERS:")
    for i, (_, row) in enumerate(df_precision_sorted.head(5).iterrows(), 1):
        print(f"  {i}. {row['segment']}/{row['mode']} (max_depth={row['max_depth']})")
        print(f"     - Precision: {row['precision']:.3f}")
        print(f"     - Recall: {row['recall']:.3f}")
        print(f"     - F1: {row['f1_score']:.3f}")
    
    print(f"\n📊 PRECISION STATISTICS:")
    print(f"  Mean Precision: {df['precision'].mean():.3f}")
    print(f"  Median Precision: {df['precision'].median():.3f}")
    print(f"  Min Precision: {df['precision'].min():.3f}")
    print(f"  Max Precision: {df['precision'].max():.3f}")
    
    # Models with precision < 0.6
    low_precision = df[df['precision'] < 0.6]
    if len(low_precision) > 0:
        print(f"\n⚠️  MODELS WITH LOW PRECISION (< 0.6):")
        for _, row in low_precision.iterrows():
            print(f"  - {row['segment']}/{row['mode']}: {row['precision']:.3f} precision")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    # Find best and worst performers
    best_overall = df.loc[df['f1_score'].idxmax()]
    worst_overall = df.loc[df['f1_score'].idxmin()]
    
    print(f"\n🏆 BEST OVERALL PERFORMER:")
    print(f"  {best_overall['segment']}/{best_overall['mode']} (F1: {best_overall['f1_score']:.3f})")
    print(f"  - Precision: {best_overall['precision']:.3f}")
    print(f"  - Recall: {best_overall['recall']:.3f}")
    
    print(f"\n❌ WORST OVERALL PERFORMER:")
    print(f"  {worst_overall['segment']}/{worst_overall['mode']} (F1: {worst_overall['f1_score']:.3f})")
    print(f"  - Precision: {worst_overall['precision']:.3f}")
    print(f"  - Recall: {worst_overall['recall']:.3f}")
    
    # Recommendations based on analysis
    print(f"\n💡 RECOMMENDATIONS:")
    
    if df['precision'].mean() < 0.7:
        print("  - Overall precision is moderate. Consider:")
        print("    * Feature engineering for better predictive power")
        print("    * Hyperparameter tuning (learning rate, max_depth)")
        print("    * Different algorithms (Random Forest, SVM)")
    
    if df['recall'].mean() < 0.6:
        print("  - Overall recall is low. Model is missing many positive cases.")
        print("    Consider adjusting class weights or threshold tuning.")
    
    # Precision vs Recall trade-off analysis
    high_precision_low_recall = df[(df['precision'] > 0.8) & (df['recall'] < 0.6)]
    if len(high_precision_low_recall) > 0:
        print("  - Some models have high precision but low recall:")
        for _, row in high_precision_low_recall.iterrows():
            print(f"    * {row['segment']}/{row['mode']}: P={row['precision']:.3f}, R={row['recall']:.3f}")
        print("    Consider threshold tuning to improve recall.")
    
    # Segment-specific recommendations
    airport_data = df[df['segment'] == 'airport']
    churned_data = df[df['segment'] == 'churned']
    
    if len(airport_data) > 0 and len(churned_data) > 0:
        if airport_data['f1_score'].mean() > churned_data['f1_score'].mean():
            print("  - Airport segment performs better than churned segment.")
            print("    Focus on airport-specific features and patterns.")
        else:
            print("  - Churned segment performs better than airport segment.")
            print("    Focus on churned rider-specific features and patterns.")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    # Analyze all reports
    df = analyze_all_reports()
    
    if len(df) > 0:
        print_detailed_analysis(df)
        create_analysis_plots(df)
        
        # Save the analysis to CSV
        df.to_csv('reports/positive_class_analysis_summary.csv', index=False)
        print(f"\n📊 Analysis saved to 'reports/positive_class_analysis_summary.csv'")
        print(f"📈 Plots saved to 'plots/positive_class_analysis_summary.pdf' and '.png'")
    else:
        print("No classification reports found to analyze.") 