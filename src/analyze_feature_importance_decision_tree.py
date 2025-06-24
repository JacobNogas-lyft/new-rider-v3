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

def analyze_decision_tree_feature_importance():
    """Analyze feature importance across all saved decision tree models (all_features only)."""
    all_results = []
    for root, _, files in os.walk('models/decision_tree'):
        for file in files:
            if file != 'decision_tree_model.joblib':
                continue
            model_path = Path(os.path.join(root, file))
            path_parts = model_path.parts
            if 'all_features' not in path_parts:
                continue
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

def create_feature_importance_summary_decision_tree(df):
    """Create a grid of subplots for decision tree feature importance by (segment, mode)."""
    if df is None:
        return
    plt.style.use('default')
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
    plt.suptitle('Top 20 Decision Tree Feature Importances by Segment and Mode', fontsize=18, y=1.02)
    plt.savefig('plots/feature_importance_by_segment_mode_decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('plots/feature_importance_by_segment_mode_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n📊 Summary grid plot saved to 'plots/feature_importance_by_segment_mode_decision_tree.pdf' and '.png'")

if __name__ == "__main__":
    Path('plots').mkdir(exist_ok=True)
    df = analyze_decision_tree_feature_importance()
    if df is not None:
        create_feature_importance_summary_decision_tree(df) 