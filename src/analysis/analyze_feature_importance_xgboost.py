import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os
import joblib

def extract_decision_tree_feature_importance(model_path):
    """Load a saved decision tree model and extract feature importances and names."""
    try:
        clf = joblib.load(model_path)
        # Try to get feature names from the model if available
        if hasattr(clf, 'feature_names_in_'):
            feature_names = clf.feature_names_in_
        else:
            # Fallback: try to infer from directory structure or skip
            feature_names = [f'feature_{i}' for i in range(len(clf.feature_importances_))]
        importances = clf.feature_importances_
        return feature_names, importances
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, None

def analyze_decision_tree_feature_importance():
    """Analyze feature importance across all saved decision tree models (all_features only)."""
    all_results = []
    
    # Walk through the models directory
    for root, _, files in os.walk('../models/decision_tree'):
        for file in files:
            if file != 'decision_tree_model.joblib':
                continue
            model_path = Path(os.path.join(root, file))
            path_parts = model_path.parts
            # Only process models in all_features directory
            if 'all_features' not in path_parts:
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
            feature_names, importances = extract_decision_tree_feature_importance(model_path)
            if feature_names is not None and importances is not None:
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances,
                    'segment': segment,
                    'mode': mode
                })
                all_results.append(df)
    if not all_results:
        print("No feature importance data found from saved models.")
        return None
    return pd.concat(all_results, ignore_index=True)

def create_feature_importance_summary(df):
    """Create separate feature importance plots for each segment."""
    if df is None:
        return
    
    plt.style.use('default')
    
    # Create summary subdirectory
    Path('../plots/summary').mkdir(exist_ok=True)
    
    # Create separate plots for each segment
    for segment in df['segment'].unique():
        segment_df = df[df['segment'] == segment]
        
        # Get unique modes for this segment
        modes = segment_df['mode'].unique()
        n_modes = len(modes)
        
        # Create subplot grid for this segment
        ncols = 2
        nrows = int(np.ceil(n_modes / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6))
        
        # Handle single subplot case
        if n_modes == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        # Create subplot for each mode in this segment
        for idx, mode in enumerate(modes):
            mode_data = segment_df[segment_df['mode'] == mode]
            
            # Get top 15 features for this mode
            top_features = mode_data.sort_values('importance', ascending=False).head(15)
            
            ax = axes[idx]
            bars = ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], 
                          color='forestgreen', alpha=0.8, edgecolor='darkgreen')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'][::-1])):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'{mode.upper()} Mode\nXGBoost Feature Importance - {segment.upper()} Segment', fontsize=13, fontweight='bold')
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(axis='x', alpha=0.3)
            
            # Set x-axis limit to accommodate labels
            max_importance = top_features['importance'].max()
            ax.set_xlim(0, max_importance * 1.15)
        
        # Remove empty subplots
        for j in range(len(modes), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        # Save segment-specific plot
        segment_filename = f'xgboost_feature_importance_{segment}_segment.png'
        plt.savefig(f'../plots/summary/{segment_filename}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved {segment_filename}")
    
    # Also create the original combined plot for reference
    pairs = list(df.groupby(['segment', 'mode']).groups.keys())
    n_pairs = len(pairs)
    ncols = 3
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 7))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, ((segment, mode), group) in enumerate(df.groupby(['segment', 'mode'])):
        top_features = group.sort_values('importance', ascending=False).head(20)
        ax = axes[idx]
        ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], color='forestgreen')
        ax.set_xlabel('Importance')
        ax.set_title(f'Segment: {segment}\nMode: {mode}')
        ax.tick_params(axis='y', labelsize=8)
    
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle('Top 20 XGBoost Feature Importances by Segment and Mode', fontsize=18, y=1.02)
    plt.savefig('../plots/feature_importance_by_segment_mode_xgboost.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/feature_importance_by_segment_mode_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n📊 Summary grid plot saved to '../plots/feature_importance_by_segment_mode_xgboost.pdf' and '.png'")
    print("📊 Segment-specific plots saved to '../plots/summary/xgboost_feature_importance_<segment>_segment.png'")

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    Path('../plots').mkdir(exist_ok=True)
    # Analyze feature importance from saved decision tree models
    df = analyze_decision_tree_feature_importance()
    if df is not None:
        create_feature_importance_summary(df) 