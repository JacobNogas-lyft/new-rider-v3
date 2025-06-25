import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def parse_classification_report(report_path):
    """Parse a classification report and extract metrics for premium class."""
    try:
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Extract metrics for "premium (not preselected)" class
        lines = content.strip().split('\n')
        for line in lines:
            if line.strip().startswith('premium (not preselected)'):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        precision = float(parts[-4])
                        recall = float(parts[-3])
                        f1_score = float(parts[-2])
                        support = int(parts[-1])
                        return {
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1_score,
                            'support': support
                        }
                    except (ValueError, IndexError):
                        continue
        return None
    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        return None

def extract_config_from_path(path):
    """Extract configuration parameters from the file path."""
    path_str = str(path)
    
    # Extract segment type
    segment_match = re.search(r'segment_(\w+)', path_str)
    segment = segment_match.group(1) if segment_match else 'unknown'
    
    # Extract mode
    mode_match = re.search(r'mode_(\w+)', path_str)
    mode = mode_match.group(1) if mode_match else 'unknown'
    
    # Extract max_depth
    depth_match = re.search(r'max_depth_(\w+)', path_str)
    max_depth = depth_match.group(1) if depth_match else 'unknown'
    
    # Extract pruning parameters
    split_match = re.search(r'split(\d+)', path_str)
    leaf_match = re.search(r'leaf(\d+)', path_str)
    ccp_match = re.search(r'ccp([\d.]+)', path_str)
    
    min_samples_split = int(split_match.group(1)) if split_match else None
    min_samples_leaf = int(leaf_match.group(1)) if leaf_match else None
    ccp_alpha = float(ccp_match.group(1)) if ccp_match else None
    
    return {
        'segment': segment,
        'mode': mode,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'ccp_alpha': ccp_alpha
    }

def collect_all_results():
    """Collect all classification report results."""
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/all_features')
    results = []
    
    print(f"Searching in: {base_path}")
    
    # Find all classification report files
    report_files = list(base_path.rglob('classification_report.txt'))
    print(f"Found {len(report_files)} classification report files")
    
    for report_file in report_files:
        print(f"Processing: {report_file}")
        config = extract_config_from_path(report_file)
        print(f"  Config: {config}")
        metrics = parse_classification_report(report_file)
        print(f"  Metrics: {metrics}")
        
        if config and metrics:
            result = {**config, **metrics}
            results.append(result)
            print(f"  Added result: {result}")
        else:
            print(f"  Skipped - config: {config is not None}, metrics: {metrics is not None}")
    
    print(f"Total results collected: {len(results)}")
    return results

def analyze_premium_performance():
    """Analyze premium class performance across all configurations."""
    print("Collecting results from all classification reports...")
    results = collect_all_results()
    
    if not results:
        print("No results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter for premium mode only
    premium_df = df[df['mode'] == 'premium'].copy()
    
    if premium_df.empty:
        print("No premium mode results found!")
        return
    
    print(f"\nFound {len(premium_df)} premium mode configurations")
    print("\nConfiguration summary:")
    print(f"Segments: {premium_df['segment'].unique()}")
    print(f"Max depths: {premium_df['max_depth'].unique()}")
    print(f"Min samples split: {premium_df['min_samples_split'].unique()}")
    print(f"Min samples leaf: {premium_df['min_samples_leaf'].unique()}")
    print(f"CCP alpha: {premium_df['ccp_alpha'].unique()}")
    
    # Sort by performance metrics
    premium_df_sorted = premium_df.sort_values(['f1_score', 'precision', 'recall'], ascending=False)
    
    print("\n" + "="*80)
    print("PREMIUM CLASS PERFORMANCE RANKING (by F1-score)")
    print("="*80)
    
    for idx, row in premium_df_sorted.iterrows():
        config_str = f"segment={row['segment']}, depth={row['max_depth']}, split={row['min_samples_split']}, leaf={row['min_samples_leaf']}, ccp={row['ccp_alpha']}"
        print(f"\n{config_str}")
        print(f"  Precision: {row['precision']:.3f}")
        print(f"  Recall:    {row['recall']:.3f}")
        print(f"  F1-score:  {row['f1_score']:.3f}")
        print(f"  Support:   {row['support']}")
    
    # Create visualizations
    create_performance_plots(premium_df)
    
    # Save detailed results
    output_path = Path('/home/sagemaker-user/studio/src/new-rider-v3/reports/premium_class_analysis.csv')
    premium_df_sorted.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return premium_df_sorted

def create_performance_plots(df):
    """Create visualizations of the performance metrics."""
    print("\nCreating performance visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Premium Class Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. F1-score by configuration
    ax1 = axes[0, 0]
    config_labels = [f"{row['segment']}_{row['max_depth']}_split{row['min_samples_split']}_leaf{row['min_samples_leaf']}_ccp{row['ccp_alpha']}" 
                    for _, row in df.iterrows()]
    ax1.bar(range(len(df)), df['f1_score'])
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score by Configuration')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall scatter plot
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['precision'], df['recall'], c=df['f1_score'], 
                         s=100, alpha=0.7, cmap='viridis')
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('Precision vs Recall (colored by F1-score)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1-Score')
    
    # 3. Performance by segment
    ax3 = axes[1, 0]
    segment_metrics = df.groupby('segment')[['precision', 'recall', 'f1_score']].mean()
    segment_metrics.plot(kind='bar', ax=ax3)
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Score')
    ax3.set_title('Average Performance by Segment')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Performance by max_depth
    ax4 = axes[1, 1]
    depth_metrics = df.groupby('max_depth')[['precision', 'recall', 'f1_score']].mean()
    depth_metrics.plot(kind='bar', ax=ax4)
    ax4.set_xlabel('Max Depth')
    ax4.set_ylabel('Score')
    ax4.set_title('Average Performance by Max Depth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path('/home/sagemaker-user/studio/src/new-rider-v3/plots/premium_class_performance_analysis.png')
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to: {plot_path}")
    plt.show()

def find_best_configuration(df):
    """Find the best configuration based on different criteria."""
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY DIFFERENT CRITERIA")
    print("="*80)
    
    # Best by F1-score
    best_f1 = df.loc[df['f1_score'].idxmax()]
    print(f"\nBest by F1-score:")
    print(f"  Config: segment={best_f1['segment']}, depth={best_f1['max_depth']}, split={best_f1['min_samples_split']}, leaf={best_f1['min_samples_leaf']}, ccp={best_f1['ccp_alpha']}")
    print(f"  F1-score: {best_f1['f1_score']:.3f}")
    
    # Best by Precision
    best_precision = df.loc[df['precision'].idxmax()]
    print(f"\nBest by Precision:")
    print(f"  Config: segment={best_precision['segment']}, depth={best_precision['max_depth']}, split={best_precision['min_samples_split']}, leaf={best_precision['min_samples_leaf']}, ccp={best_precision['ccp_alpha']}")
    print(f"  Precision: {best_precision['precision']:.3f}")
    
    # Best by Recall
    best_recall = df.loc[df['recall'].idxmax()]
    print(f"\nBest by Recall:")
    print(f"  Config: segment={best_recall['segment']}, depth={best_recall['max_depth']}, split={best_recall['min_samples_split']}, leaf={best_recall['min_samples_leaf']}, ccp={best_recall['ccp_alpha']}")
    print(f"  Recall: {best_recall['recall']:.3f}")
    
    # Best balanced (closest to 1.0 for all metrics)
    df['balanced_score'] = ((1 - df['precision']) + (1 - df['recall']) + (1 - df['f1_score'])) / 3
    best_balanced = df.loc[df['balanced_score'].idxmin()]
    print(f"\nBest Balanced (closest to 1.0 for all metrics):")
    print(f"  Config: segment={best_balanced['segment']}, depth={best_balanced['max_depth']}, split={best_balanced['min_samples_split']}, leaf={best_balanced['min_samples_leaf']}, ccp={best_balanced['ccp_alpha']}")
    print(f"  Precision: {best_balanced['precision']:.3f}, Recall: {best_balanced['recall']:.3f}, F1: {best_balanced['f1_score']:.3f}")

if __name__ == "__main__":
    print("Analyzing premium class performance across all configurations...")
    results_df = analyze_premium_performance()
    
    if results_df is not None:
        find_best_configuration(results_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("Check the generated CSV file and plots for detailed analysis.") 