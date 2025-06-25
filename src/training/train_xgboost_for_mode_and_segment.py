import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from utils.load_data import load_parquet_data
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib

# Configuration parameters

# Categorical columns to drop (high cardinality features that cause memory issues)
CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                            'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id']

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

def load_and_prepare_data(segment_type='all'):
    print(f"Loading and preparing data for segment: {segment_type}...")
    df = load_parquet_data()
    df = add_churned_indicator(df)
    
    # Filter by segment
    df = filter_by_segment(df, segment_type)
    
    required_cols = ['requested_ride_type', 'preselected_mode']
    df = df.dropna(subset=required_cols)
    return df

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
    print(f"  - percent_rides_standard_lifetime: {df['percent_rides_standard_lifetime'].describe()}")
    print(f"  - percent_rides_premium_lifetime: {df['percent_rides_premium_lifetime'].describe()}")
    print(f"  - percent_rides_plus_lifetime: {df['percent_rides_plus_lifetime'].describe()}")
    
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Use all categorical columns except the high cardinality ones
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

def train_xgboost(X_train, y_train, max_depth):
    print(f"Training XGBoost (max_depth={max_depth})...")
    
    # XGBoost parameters
    params = {
        'max_depth': max_depth,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'scale_pos_weight': 1.0,  # Will be calculated based on class imbalance
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    if pos_count > 0:
        params['scale_pos_weight'] = neg_count / pos_count
    
    # Train XGBoost model
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    print("Model training complete.")
    return clf

def save_model(clf, models_dir):
    """Save the trained model and its feature names."""
    print("Saving model...")
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / 'xgboost_model.joblib'
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

def evaluate_model(clf, X_test, y_test, class_names, reports_dir):
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    reports_dir.mkdir(exist_ok=True, parents=True)
    report_path = reports_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
        f.write(f"\nPrediction probabilities summary:\n")
        f.write(f"Min probability: {y_pred_proba.min():.4f}\n")
        f.write(f"Max probability: {y_pred_proba.max():.4f}\n")
        f.write(f"Mean probability: {y_pred_proba.mean():.4f}\n")
        f.write(f"Std probability: {y_pred_proba.std():.4f}\n")
    print(f"Saved classification report to {report_path}")

def plot_feature_importance(clf, X, plots_dir):
    print("Plotting feature importances...")
    
    # Create the plots directory if it doesn't exist
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 20
    
    plt.figure(figsize=(16, 8))
    plt.title('XGBoost Feature Importances (Top 20)', fontsize=20)
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [X.columns[i] for i in indices[:top_n]], rotation=45, ha='right', fontsize=12)
    plt.ylabel('Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.pdf')
    print(f"Saved feature importance plot to {plots_dir / 'feature_importance.pdf'}")
    
    # Also save as SVG for better quality
    plt.savefig(plots_dir / 'feature_importance.svg')
    print(f"Saved feature importance plot to {plots_dir / 'feature_importance.svg'}")
    
    # Close the plot to free memory
    plt.close()

def run_for_mode_and_depth(df, mode, max_depth, segment_type):
    print(f"\n=== Running for segment: {segment_type}, mode: {mode}, max_depth: {max_depth} ===")
    X, y, feature_cols = prepare_features_and_target(df, mode)
    # If only one class, skip
    if y.nunique() < 2:
        print(f"Skipping segment={segment_type}, mode={mode}, max_depth={max_depth}: only one class present.")
        return
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_xgboost(X_train, y_train, max_depth)
    class_names = [f'not {mode}', f'{mode} (not preselected)']
    
    # Define directories
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    plots_dir = base_path / f'plots/xg_boost/all_features/segment_{segment_type}/mode_{mode}/max_depth_{max_depth}'
    reports_dir = base_path / f'reports/xg_boost/all_features/segment_{segment_type}/mode_{mode}/max_depth_{max_depth}'
    models_dir = base_path / f'models/xg_boost/all_features/segment_{segment_type}/mode_{mode}/max_depth_{max_depth}'
    
    # Save model, evaluate, and create plots
    save_model(clf, models_dir)
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
    plot_feature_importance(clf, X, plots_dir)

def main(mode_list, max_depth_list, segment_type_list):
    print(f"Training XGBoost models for segments: {segment_type_list}")
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
        
        # Create base directories
        base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
        (base_path / 'models/xg_boost/all_features').mkdir(exist_ok=True, parents=True)
        (base_path / 'plots/xg_boost/all_features').mkdir(exist_ok=True, parents=True)
        (base_path / 'reports/xg_boost/all_features').mkdir(exist_ok=True, parents=True)
        
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
        print(f"  - Models: {base_path}/models/xg_boost/all_features/segment_{segment_type}/")
        print(f"  - Plots: {base_path}/plots/xg_boost/all_features/segment_{segment_type}/")
        print(f"  - Reports: {base_path}/reports/xg_boost/all_features/segment_{segment_type}/")

if __name__ == "__main__":
    # Configuration
    segment_type_list = ['churned', 'airport', 'all']
    
    # Max depth list - simplified
    max_depth_list = [10]  # Simple depth values
    
    mode_list = ['fastpass', 'standard', 'premium', 'plus', 'lux', 'luxsuv']
    
    main(mode_list, max_depth_list, segment_type_list) 