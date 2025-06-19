import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

def parse_feature_importance(file_path):
    """Parse feature importance data from a PDF file name."""
    # Since we can't read PDF directly, we'll look for a corresponding data file
    data_file = file_path.with_suffix('.csv')
    if not data_file.exists():
        return None
    
    df = pd.read_csv(data_file)
    return df

def analyze_feature_importance():
    """Analyze feature importance across all XGBoost models with max_depth_10."""
    all_results = []
    
    # Walk through the plots directory
    for root, _, files in os.walk('plots/xg_boost'):
        for file in files:
            if file != 'feature_importance.pdf':
                continue
            
            file_path = Path(os.path.join(root, file))
            path_parts = file_path.parts
            
            # Skip if not max_depth_10
            if 'max_depth_10' not in path_parts:
                continue
            
            # Extract segment and mode from path
            segment = None
            mode = None
            
            for part in path_parts:
                if part.startswith('segment_'):
                    segment = part.replace('segment_', '')
                elif part.startswith('mode_'):
                    mode = part.replace('mode_', '')
            
            if not all([segment, mode]):
                continue
            
            # Parse feature importance data
            importance_data = parse_feature_importance(file_path)
            if importance_data is not None:
                importance_data['segment'] = segment
                importance_data['mode'] = mode
                all_results.append(importance_data)
    
    if not all_results:
        print("No feature importance data found.")
        return None
    
    return pd.concat(all_results, ignore_index=True)

def create_feature_importance_summary(df):
    """Create visualization plots for feature importance analysis."""
    if df is None:
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 25))
    
    # 1. Overall Feature Importance
    plt.subplot(3, 1, 1)
    overall_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=True)
    plt.barh(range(len(overall_importance)), overall_importance.values)
    plt.yticks(range(len(overall_importance)), overall_importance.index)
    plt.title('Average Feature Importance Across All Models', fontsize=14, pad=20)
    plt.xlabel('Mean Importance')
    
    # 2. Feature Importance Heatmap by Mode and Segment
    plt.subplot(3, 1, 2)
    pivot_importance = df.pivot_table(
        index='feature',
        columns=['segment', 'mode'],
        values='importance',
        aggfunc='mean'
    )
    sns.heatmap(pivot_importance, cmap='YlOrRd', center=0)
    plt.title('Feature Importance by Mode and Segment', fontsize=14, pad=20)
    
    # 3. Top Features Distribution
    plt.subplot(3, 1, 3)
    top_features = overall_importance.tail(10).index
    top_features_data = df[df['feature'].isin(top_features)]
    
    sns.boxplot(data=top_features_data, x='importance', y='feature')
    plt.title('Distribution of Top 10 Feature Importance', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig('plots/feature_importance_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('plots/feature_importance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n=== Feature Importance Analysis ===")
    print("\nTop 10 Most Important Features (Average):")
    for feature, importance in overall_importance.tail(10).items():
        print(f"{feature}: {importance:.4f}")
    
    print("\nFeature Importance Variation:")
    importance_std = df.groupby('feature')['importance'].std().sort_values(ascending=False)
    print("\nFeatures with Highest Variation:")
    for feature, std in importance_std.head(5).items():
        print(f"{feature}: std={std:.4f}")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    # Analyze feature importance
    df = analyze_feature_importance()
    
    if df is not None:
        create_feature_importance_summary(df)
        print("\n📊 Summary plots saved to 'plots/feature_importance_summary.pdf' and '.png'") 