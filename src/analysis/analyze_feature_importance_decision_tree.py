import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import joblib

def extract_decision_tree_feature_importance(model_path):
    """Load a saved decision tree model and extract feature importances and names."""
    try:
        clf = joblib.load(model_path)
        if hasattr(clf, 'feature_names_in_'):
            feature_names = clf.feature_names_in_
        else:
            feature_names = [f'feature_{i}' for i in range(len(clf.feature_importances_))]
        importances = clf.feature_importances_
        return feature_names, importances
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, None

def analyze_decision_tree_feature_importance(data_version='v3'):
    """Analyze feature importance across all saved decision tree models (all_features only, max_depth=10)."""
    # Determine the base path based on data_version
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    base_models_path = f'../models/decision_tree/all_features{data_suffix}'
    
    print(f"Analyzing feature importance from models in: {base_models_path}")
    print(f"Data version: {data_version.upper()}")
    print(f"Using only max_depth=10 models")
    
    # Check if directory exists
    if not os.path.exists(base_models_path):
        print(f"ERROR: Directory {base_models_path} does not exist!")
        return None
    
    all_results = []
    for root, _, files in os.walk(base_models_path):
        for file in files:
            if file != 'decision_tree_model.joblib':
                continue
            model_path = Path(os.path.join(root, file))
            path_parts = model_path.parts
            
            # Only include models in all_features directory (with or without _v2/_v3 suffix)
            all_features_found = False
            for part in path_parts:
                if 'all_features' in part:
                    all_features_found = True
                    break
            
            if not all_features_found:
                continue
            
            # Only include max_depth 10 models
            max_depth_found = False
            for part in path_parts:
                if part.startswith('max_depth_'):
                    depth_value = part.replace('max_depth_', '')
                    if depth_value == '10':
                        max_depth_found = True
                        break
            
            if not max_depth_found:
                continue
                
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
            feature_names, importances = extract_decision_tree_feature_importance(model_path)
            if feature_names is not None and importances is not None:
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances,
                    'segment': segment,
                    'mode': mode,
                    'max_depth': int(depth),
                    'data_version': data_version.upper()
                })
                all_results.append(df)
    if not all_results:
        print("No feature importance data found from saved models.")
        return None
    return pd.concat(all_results, ignore_index=True)

def create_feature_importance_summary_decision_tree(df, data_version='v3'):
    """Create separate feature importance plots for each segment.
    
    Note: Only uses models with max_depth=10 for consistency.
    """
    if df is None:
        return
    
    plt.style.use('default')
    
    # Create summary subdirectory with data version subfolder
    summary_dir = Path(f'../plots/summary/{data_version}')
    summary_dir.mkdir(exist_ok=True, parents=True)
    
    # Get data version for file naming
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    data_version_title = data_version.upper()
    
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
                          color='royalblue', alpha=0.8, edgecolor='navy')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'][::-1])):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'{mode.upper()} Mode\nDecision Tree Feature Importance - {segment.upper()} Segment\n{data_version_title} Data', fontsize=13, fontweight='bold')
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
        segment_filename = f'decision_tree_feature_importance_{segment}_segment{data_suffix}.png'
        plt.savefig(summary_dir / segment_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved {segment_filename} to {summary_dir}")
    
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
        ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], color='royalblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Segment: {segment}\nMode: {mode}')
        ax.tick_params(axis='y', labelsize=8)
    
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle(f'Top 20 Decision Tree Feature Importances by Segment and Mode - {data_version_title} Data', fontsize=18, y=1.02)
    plt.savefig(summary_dir / f'feature_importance_by_segment_mode_decision_tree{data_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(summary_dir / f'feature_importance_by_segment_mode_decision_tree{data_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Summary grid plot saved to '{summary_dir}/feature_importance_by_segment_mode_decision_tree{data_suffix}.pdf' and '.png'")
    print(f"📊 Segment-specific plots saved to '{summary_dir}/decision_tree_feature_importance_<segment>_segment{data_suffix}.png'")

if __name__ == "__main__":
    # Set data version: 'original', 'v2', or 'v3'
    data_version = 'v3'
    
    Path('../plots').mkdir(exist_ok=True)
    print(f"Starting Decision Tree feature importance analysis with {data_version.upper()} data...")
    df = analyze_decision_tree_feature_importance(data_version)
    if df is not None:
        create_feature_importance_summary_decision_tree(df, data_version)
        print(f"✅ Feature importance analysis complete for {data_version.upper()} data")
    else:
        print(f"❌ No models found for {data_version.upper()} data") 