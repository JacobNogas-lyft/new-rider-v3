import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from utils.load_data import load_parquet_data
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib


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
    'favorite_product_key_28d', 'favorite_product_key_90d',
    
    # Airline feature (will be added during processing)
    'airline'
]

CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                            'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id',
                            'rider_lyft_id',
                            'signup_at',
                            'destination_place_name',
                            'pickup_place_name'
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
            - 'airport_dropoff': Sessions where destination_venue_category = 'airport'
            - 'airport_pickup': Sessions where origin_venue_category = 'airport'
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
    
    elif segment_type == 'airport_dropoff':
        airport_dropoff_mask = (df['destination_venue_category'] == 'airport')
        filtered_df = df[airport_dropoff_mask].copy()
        print(f"Airport dropoff sessions: {len(filtered_df)} rows (from {len(df)} total)")
    
    elif segment_type == 'airport_pickup':
        airport_pickup_mask = (df['origin_venue_category'] == 'airport')
        filtered_df = df[airport_pickup_mask].copy()
        print(f"Airport pickup sessions: {len(filtered_df)} rows (from {len(df)} total)")
    
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
        raise ValueError(f"Unknown segment type: {segment_type}. Use 'airport', 'airport_dropoff', 'airport_pickup', 'churned', or 'all'")
    
    return filtered_df

def prepare_features_and_target(df, mode):
    df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == mode)).astype(int)
    
    # Add percentage features for lifetime rides
    print("Creating percentage features for lifetime rides...")
    
    # Handle division by zero by replacing 0 with NaN, then filling with 0
    df['percent_rides_standard_lifetime'] = (
        df['rides_standard_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    df['percent_rides_premium_lifetime'] = (
        df['rides_premium_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    df['percent_rides_plus_lifetime'] = (
        df['rides_plus_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    print(f"Created percentage features:")
    
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

def save_model(clf, models_dir):
    """Save the trained decision tree model and its feature names."""
    print("Saving model...")
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / 'decision_tree_model.joblib'
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

def run_for_mode_and_depth(df, mode, max_depth, segment_type, data_version='original'):
    print(f"\n=== Running for segment: {segment_type}, mode: {mode}, max_depth: {max_depth}, data_version: {data_version} ===")
    
    X, y, feature_cols = prepare_features_and_target(df, mode)
    # If only one class, skip
    if y.nunique() < 2:
        print(f"Skipping segment={segment_type}, mode={mode}, max_depth={max_depth}, data_version={data_version}: only one class present.")
        return
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_decision_tree(X_train, y_train, max_depth)
    class_names = [f'not {mode}', f'{mode} (not preselected)']
    
    # Create directory names with data version suffix
    max_depth_str = str(max_depth) if max_depth is not None else "unbounded"
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    plots_dir = base_path / f'plots/decision_tree/all_features{data_suffix}/segment_{segment_type}/mode_{mode}/max_depth_{max_depth_str}'
    reports_dir = base_path / f'reports/decision_tree/all_features{data_suffix}/segment_{segment_type}/mode_{mode}/max_depth_{max_depth_str}'
    models_dir = base_path / f'models/decision_tree/all_features{data_suffix}/segment_{segment_type}/mode_{mode}/max_depth_{max_depth_str}'
    
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
    visualize_tree(clf, X, class_names, plots_dir, max_depth)
    plot_feature_importance(clf, X, plots_dir)
    save_model(clf, models_dir)

def main(mode_list, max_depth_list, segment_type_list, data_version='original'):
    print(f"Training decision trees for segments: {segment_type_list}")
    print(f"Modes: {mode_list}")
    print(f"Max depths: {max_depth_list}")
    print(f"Using {data_version.upper()} data")
    print("="*60)
    
    # Load data once
    print("Loading data...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    

    
    # Select one row per purchase_session_id arbitrarily
    if data_version in ['v2', 'v3']:
        assert 'rider_lyft_id' in df.columns, f"rider_lyft_id should be in columns when data_version={data_version}, but not found. Available columns: {[col for col in df.columns if 'session' in col]}"
        print(f"Verified: rider_lyft_id is in columns for {data_version.upper()} data")

    df = df.drop_duplicates(subset=['purchase_session_id'], keep='first')
    print(f"After deduplication: {len(df)} rows")
    # Assert that we have exactly 1 row per purchase_session_id
    assert df['purchase_session_id'].nunique() == len(df), f"Expected 1 row per purchase_session_id, but got {len(df)} rows for {df['purchase_session_id'].nunique()} unique purchase_session_ids"
    print(f"Verified: {len(df)} rows with {df['purchase_session_id'].nunique()} unique purchase_session_ids")

    df.drop(columns=CATEGORICAL_COLS_TO_DROP, inplace=True)
    # Assert that rider_session_id is not in columns
    assert 'rider_lyft_id' not in df.columns, f"rider_lyft_id should not be in columns, but found: {[col for col in df.columns if 'rider_lyft_id' in col]}"
    print(f"Verified: rider_lyft_id not in columns")

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
                        df_segment, mode, max_depth, segment_type, data_version
                    ))
            
            # Process results with better error handling
            for i, future in enumerate(as_completed(tasks)):
                try:
                    future.result()  # To raise exceptions if any
                except Exception as e:
                    print(f"Error in task {i}: {e}")
                    print("Continuing with other tasks...")
                    continue
        
        data_suffix = f"_{data_version}" if data_version != 'original' else ""
        print(f"\nTraining completed for segment: {segment_type}")
        print(f"Results saved to:")
        print(f"  - Plots: /home/sagemaker-user/studio/src/new-rider-v3/plots/decision_tree/all_features{data_suffix}/segment_{segment_type}/")
        print(f"  - Reports: /home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/all_features{data_suffix}/segment_{segment_type}/")
        print(f"  - Models: /home/sagemaker-user/studio/src/new-rider-v3/models/decision_tree/all_features{data_suffix}/segment_{segment_type}/")

if __name__ == "__main__":
    # Configuration
    #segment_type_list = ['churned', 'airport', 'airport_dropoff', 'airport_pickup', 'all']
    segment_type_list = ['airport_dropoff', 'airport_pickup']
    #segment_type_list = ['all']
    
    # Max depth list - simplified
    #max_depth_list = [10]  # Simple depth values
    max_depth_list = [3,5,10]  # Simple depth values
    
    mode_list = ['fastpass', 'standard', 'premium', 'plus', 'lux', 'luxsuv']
    
    # Set data version: 'original', 'v2', or 'v3'
    data_version = 'v3'
    
    main(mode_list, max_depth_list, segment_type_list, data_version) 