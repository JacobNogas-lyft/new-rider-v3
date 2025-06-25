import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from utils.load_data import load_parquet_data
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration parameters
SEGMENT_TYPE = 'airport'  # Options: 'airport', 'churned', 'all'
MAX_DEPTH = 5

# Original categorical features (commented out for reference)
# key_categorical_cols = ['region', 'currency', 'passenger_device', 'pax_os', 'pax_carrier']

# Expanded categorical features
KEY_CATEGORICAL_COLS = [
    # Original features
    'region', 'currency', 'passenger_device', 'pax_os', 'pax_carrier',
    
    # Venue/Location related
    'origin_venue_category', 'destination_venue_category',
    'place_category_pickup', 'place_category_destination',
    
    # Availability related
    'standard_availability_caveat', 'plus_availability_caveat',
    'premium_availability_caveat', 'lux_availability_caveat',
    'luxsuv_availability_caveat', 'fastpass_availability_caveat',
    'standard_saver_availability_caveat', 'green_availability_caveat',
    'pet_availability_caveat',
    
    # Payment related
    'card_issuer',
    
    # Weather related
    'forecast_hr_gh4_precip_type', 'forecast_hr_gh4_summary',
    
    # Historical mode preferences
    'last_product_key', 'second_last_product_key', 'third_last_product_key',
    'fourth_last_product_key', 'fifth_last_product_key',
    'favorite_product_key_28d', 'favorite_product_key_90d'
]

CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                            'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id'
                            ]#what is label?
                            #['bundle_set_id', 'rider_session_id', 'occurred_at', 'requested_at', 'candidate_product_keylabel']

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

def prepare_features_and_target(df, mode):
    df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == mode)).astype(int)
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    feature_cols = [col for col in numeric_cols if col not in drop_cols] + [col for col in categorical_cols if (col not in drop_cols) and (col not in CATEGORICAL_COLS_TO_DROP)]
    X = df[feature_cols]
    y = df['target_diff_mode']
    
    print(f"Before get_dummies - X shape: {X.shape}")
    print(f"Memory usage before encoding: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"After get_dummies - X shape: {X.shape}")
    print(f"Memory usage after encoding: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("Data loaded and processed.")
    return X, y, feature_cols

def split_data(X, y):
    print("Splitting data into train and test sets (stratified)...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def downsample_training_data(X_train, y_train, ratio=2):
    """
    Downsample only the training data by randomly sampling from the majority class.
    
    Args:
        X_train (pandas.DataFrame): Training feature matrix
        y_train (pandas.Series): Training target variable
        ratio (int): Ratio of majority to minority class (e.g., 2 means 1:2 ratio)
    
    Returns:
        tuple: (X_train_downsampled, y_train_downsampled)
    """
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    print(f"Original training class distribution: {y_train.value_counts().to_dict()}")
    
    # Get minority and majority class counts
    class_counts = y_train.value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    minority_count = class_counts[minority_class]
    majority_count = class_counts[majority_class]
    
    # Calculate target majority count based on ratio
    target_majority_count = minority_count * ratio
    
    print(f"Minority class ({minority_class}): {minority_count}")
    print(f"Majority class ({majority_class}): {majority_count}")
    print(f"Target majority count for 1:{ratio} ratio: {target_majority_count}")
    
    # Get indices for each class
    minority_indices = y_train[y_train == minority_class].index
    majority_indices = y_train[y_train == majority_class].index
    
    # Randomly sample from majority class
    if len(majority_indices) > target_majority_count:
        sampled_majority_indices = np.random.choice(
            majority_indices, 
            size=int(target_majority_count), 
            replace=False
        )
    else:
        # If majority class is smaller than target, use all samples
        sampled_majority_indices = majority_indices
        print(f"Warning: Majority class has fewer samples than target. Using all {len(majority_indices)} samples.")
    
    # Combine indices
    downsampled_indices = np.concatenate([minority_indices, sampled_majority_indices])
    
    # Shuffle the indices
    np.random.shuffle(downsampled_indices)
    
    # Create downsampled training dataset
    X_train_downsampled = X_train.loc[downsampled_indices]
    y_train_downsampled = y_train.loc[downsampled_indices]
    
    print(f"Downsampled training class distribution: {y_train_downsampled.value_counts().to_dict()}")
    print(f"Downsampled training dataset shape: {X_train_downsampled.shape}")
    
    return X_train_downsampled, y_train_downsampled

def train_decision_tree(X_train, y_train, max_depth):
    print(f"Training Decision Tree (max_depth={max_depth})...")
    
    # Decision Tree parameters - simplified to only use max_depth
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42, 
        class_weight='balanced'
    )
    
    try:
        clf.fit(X_train, y_train)
        
        # Print tree information
        n_nodes = clf.tree_.node_count
        n_leaves = clf.get_n_leaves()
        actual_depth = clf.get_depth()
        print(f"Tree built with {n_nodes} nodes, {n_leaves} leaves, and depth {actual_depth}")
        
        print("Model training complete.")
        return clf
        
    except Exception as e:
        print(f"Error training tree: {e}")
        print("Trying with limited depth...")
        
        # Fallback: try with a reasonable max_depth
        clf = DecisionTreeClassifier(
            max_depth=20,  # Reasonable fallback depth
            random_state=42, 
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        
        n_nodes = clf.tree_.node_count
        n_leaves = clf.get_n_leaves()
        actual_depth = clf.get_depth()
        print(f"Fallback tree built with {n_nodes} nodes, {n_leaves} leaves, and depth {actual_depth}")
        
        return clf

def evaluate_model(clf, X_test, y_test, class_names, reports_dir):
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=class_names)
    reports_dir.mkdir(exist_ok=True, parents=True)
    report_path = reports_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(f"Saved classification report to {report_path}")

def visualize_tree(clf, X, class_names, plots_dir, max_depth):
    print("Visualizing tree...")
    plots_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(24, 8))
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, max_depth=min(3, max_depth), fontsize=8)
    plt.title(f'Decision Tree (first 3 levels, max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'decision_tree_plot.svg')
    print(f"Saved decision tree plot to {plots_dir / 'decision_tree_plot.svg'}")
    plt.figure(figsize=(24, 24))
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, fontsize=8)
    plt.title(f'Decision Tree (full depth, max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'decision_tree_plot_full.svg')
    print(f"Saved full decision tree plot to {plots_dir / 'decision_tree_plot_full.svg'}")

def plot_feature_importance(clf, X, plots_dir):
    print("Plotting feature importances...")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 20
    plt.figure(figsize=(16, 8))
    plt.title('Feature Importances (Top 20)', fontsize=20)
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [X.columns[i] for i in indices[:top_n]], rotation=45, ha='right', fontsize=12)
    plt.ylabel('Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.pdf')
    print(f"Saved feature importance plot to {plots_dir / 'feature_importance.pdf'}")

def run_for_mode_and_depth(df, mode, max_depth, segment_type):
    print(f"\n=== Running for segment: {segment_type}, mode: {mode}, max_depth: {max_depth} ===")
    
    X, y, feature_cols = prepare_features_and_target(df, mode)
    # If only one class, skip
    if y.nunique() < 2:
        print(f"Skipping segment={segment_type}, mode={mode}, max_depth={max_depth}: only one class present.")
        return
    
    # Split data first
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Downsample only the training data
    print("Downsampling training data to 1:2 ratio...")
    X_train_downsampled, y_train_downsampled = downsample_training_data(X_train, y_train, ratio=2)
    
    # Train model on downsampled training data
    clf = train_decision_tree(X_train_downsampled, y_train_downsampled, max_depth)
    class_names = [f'not {mode}', f'{mode} (not preselected)']
    
    # Create directory names - simplified without pruning info
    max_depth_str = str(max_depth) if max_depth is not None else "unbounded"
    plots_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/decision_tree/all_features/segment_{segment_type}/mode_{mode}/max_depth_{max_depth_str}_downsample')
    reports_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/all_features/segment_{segment_type}/mode_{mode}/max_depth_{max_depth_str}_downsample')
    
    # Evaluate on original test set (not downsampled)
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
    visualize_tree(clf, X, class_names, plots_dir, max_depth)
    plot_feature_importance(clf, X, plots_dir)

def main(mode_list, max_depth_list, segment_type_list):
    print(f"Training decision trees for segments: {segment_type_list}")
    print(f"Modes: {mode_list}")
    print(f"Max depths: {max_depth_list}")
    print("="*60)
    
    # Load data once
    print("Loading data...")
    df = load_parquet_data()
    df = add_churned_indicator(df)
    df.drop(columns=CATEGORICAL_COLS_TO_DROP, inplace=True)
    
    # Process each segment type
    for segment_type in segment_type_list:
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_type}")
        print(f"{'='*60}")
        
        # Filter data for this segment
        df_segment = filter_by_segment(df, segment_type)
        
        # Filter out rows with missing required columns
        required_cols = ['requested_ride_type', 'preselected_mode']
        df_segment = df_segment.dropna(subset=required_cols)
        
        print(f"Segment data shape: {df_segment.shape}")
        
        # Parallelize using ProcessPoolExecutor
        tasks = []
        with ProcessPoolExecutor() as executor:
            for max_depth in max_depth_list:
                for mode in mode_list:
                    tasks.append(executor.submit(
                        run_for_mode_and_depth, 
                        df_segment, mode, max_depth, segment_type
                    ))
            
            # Process results with better error handling
            for i, future in enumerate(as_completed(tasks)):
                try:
                    future.result()  # To raise exceptions if any
                except Exception as e:
                    print(f"Error in task {i}: {e}")
                    print("Continuing with other tasks...")
                    continue
        
        print(f"\nTraining completed for segment: {segment_type}")
        print(f"Results saved to:")
        print(f"  - Plots: /home/sagemaker-user/studio/src/new-rider-v3/plots/decision_tree/all_features/segment_{segment_type}/")
        print(f"  - Reports: /home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/all_features/segment_{segment_type}/")

if __name__ == "__main__":
    # Configuration
    segment_type_list = ['churned', 'airport', 'all']
    
    # Max depth list - simplified
    max_depth_list = [10]  # Simple depth values
    
    mode_list = ['fastpass', 'standard', 'premium', 'plus', 'lux', 'luxsuv']
    
    main(mode_list, max_depth_list, segment_type_list) 