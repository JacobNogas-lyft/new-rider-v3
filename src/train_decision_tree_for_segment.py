import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from load_data import load_parquet_data
from pathlib import Path

# Configuration parameters
MAX_DEPTH = 5

# Create plots directory
PLOTS_DIR = Path('/home/sagemaker-user/studio/src/new-rider-v3/plots')
PLOTS_DIR.mkdir(exist_ok=True)

# --- Helper: Churned user indicator (copied from churned_rider_analysis.py) ---
def add_churned_indicator(df):
    df['ds'] = pd.to_datetime(df['ds'])
    churned_mask = (
        (df['days_since_signup'] > 365) &
        (df['rides_lifetime'] > 2) &
        (df['all_type_total_rides_365d'] == 0)
    )
    df['is_churned_user'] = churned_mask.astype(int)
    return df

def filter_by_segment(df, segment_type):
    """
    Filter dataframe by segment type.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        segment_type (str): Type of segment to filter by
            - 'airport': Sessions where destination_venue_category = 'airport' or origin_venue_category = 'airport'
            - 'churned': Sessions where rider is churned (is_churned_user = 1)
            - 'all': No filtering (use all data)
    
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    if segment_type == 'airport':
        # Filter for airport sessions
        airport_mask = (
            (df['destination_venue_category'] == 'airport') | 
            (df['origin_venue_category'] == 'airport')
        )
        filtered_df = df[airport_mask].copy()
        print(f"Airport sessions: {len(filtered_df)} rows (from {len(df)} total)")
        
    elif segment_type == 'churned':
        # Filter for churned riders
        churned_mask = (df['is_churned_user'] == 1)
        filtered_df = df[churned_mask].copy()
        print(f"Churned rider sessions: {len(filtered_df)} rows (from {len(df)} total)")
        
    elif segment_type == 'all':
        # No filtering
        filtered_df = df.copy()
        print(f"Using all data: {len(filtered_df)} rows")
        
    else:
        raise ValueError(f"Unknown segment type: {segment_type}. Use 'airport', 'churned', or 'all'")
    
    return filtered_df

# --- Main script ---
def load_and_prepare_data(segment_type='all'):
    print(f"Loading data for segment: {segment_type}...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Add churned user indicator
    df = add_churned_indicator(df)

    # Filter by segment
    df = filter_by_segment(df, segment_type)

    # Drop rows with missing required columns
    required_cols = ['requested_ride_type', 'preselected_mode']
    df = df.dropna(subset=required_cols)

    # Create target: 1 if requested_ride_type != preselected_mode, else 0
    df['target_diff_mode'] = (df['requested_ride_type'] != df['preselected_mode']).astype(int)

    # Prepare features: drop target and columns that leak target
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    key_categorical_cols = ['region', 'currency', 'passenger_device', 'pax_os', 'pax_carrier']
    feature_cols = [col for col in numeric_cols if col not in drop_cols] + key_categorical_cols
    print(f"\nSelected {len(feature_cols)} features:")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(key_categorical_cols)}")
    X = df[feature_cols]
    y = df['target_diff_mode']
    print("\nDebug info before get_dummies:")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"X shape before get_dummies: {X.shape}")
    print(f"X memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"X dtypes:\n{X.dtypes}")
    print("\nConverting categorical columns to dummy variables...")
    X = pd.get_dummies(X, drop_first=True)
    print(f"X shape after get_dummies: {X.shape}")
    print(f"X memory usage after get_dummies: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return X, y, feature_cols

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_decision_tree(X_train, y_train, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test, class_names, reports_dir):
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    reports_dir.mkdir(exist_ok=True, parents=True)
    report_path = reports_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(f"Saved classification report to {report_path}")

def visualize_tree(clf, X, class_names, plots_dir, max_depth):
    print("\nDecision Tree Structure:")
    print(export_text(clf, feature_names=list(X.columns)))
    # Visualize the tree (first 3 levels or up to max_depth)
    plots_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(24, 8))
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, max_depth=min(3, max_depth), fontsize=8)
    plt.title(f'Decision Tree (first 3 levels, max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'decision_tree_plot.svg')
    print(f"Saved decision tree plot to {plots_dir / 'decision_tree_plot.svg'}")
    # Visualize the full tree
    plt.figure(figsize=(24, 24))
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, fontsize=8)
    plt.title(f'Decision Tree (full depth, max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'decision_tree_plot_full.svg')
    print(f"Saved full decision tree plot to {plots_dir / 'decision_tree_plot_full.svg'}")

def plot_feature_importance(clf, X, plots_dir):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 20
    plt.figure(figsize=(16, 8))
    plt.title('Feature Importances (Top 20)', fontsize=20)
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [X.columns[i] for i in indices[:top_n]], rotation=45, ha='right', fontsize=12)
    plt.ylabel('Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.svg')
    print(f"Saved feature importance plot to {plots_dir / 'feature_importance.svg'}")

def main(segment_type='all', max_depth=5):
    """
    Main function to train decision tree for a specific segment.
    
    Args:
        segment_type (str): Type of segment ('airport', 'churned', or 'all')
        max_depth (int): Maximum depth of the decision tree
    """
    print(f"Training decision tree for segment: {segment_type}")
    print(f"Max depth: {max_depth}")
    print("="*60)
    
    X, y, feature_cols = load_and_prepare_data(segment_type)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_decision_tree(X_train, y_train, max_depth)
    class_names = ['requested preselected mode', 'requested non-preselected mode']
    
    # Create subfolders for this segment, max_depth and model type
    plots_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/decision_tree/segment_{segment_type}/max_depth_{max_depth}')
    reports_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/segment_{segment_type}/max_depth_{max_depth}')
    
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
    visualize_tree(clf, X, class_names, plots_dir, max_depth)
    plot_feature_importance(clf, X, plots_dir)
    
    print(f"\nTraining completed for segment: {segment_type}")
    print(f"Results saved to:")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Reports: {reports_dir}")

if __name__ == "__main__":
    segment_type = 'airport'  # Options: 'airport', 'churned', 'all'
    main(segment_type, MAX_DEPTH) 