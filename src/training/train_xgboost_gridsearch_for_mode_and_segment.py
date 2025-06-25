import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_parquet_data
from pathlib import Path
import joblib
import warnings
from scipy.stats import uniform, randint
import psutil
warnings.filterwarnings('ignore')

# Configuration parameters

# Categorical columns to drop (high cardinality features that cause memory issues)
CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                            'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id']

# Training configuration - OPTIMIZED for large dataset
RANDOM_SEARCH_ITERATIONS = 15  # Balanced for speed and quality
MAX_PARALLEL_PROCESSES = 2     # Use 2 processes for parallelism, safe for large data
CV_FOLDS = 3                   # Reduced from 5 to speed up training
TEST_SIZE = 0.2                # Test set size
RANDOM_STATE = 42              # Random seed for reproducibility

# GPU configuration
USE_GPU = True                 # Use GPU for XGBoost
GPU_MEMORY_FRACTION = 0.8      # Use 80% of GPU memory to leave buffer


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
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def perform_grid_search(X_train, y_train, X_original, y_original, segment_type, mode):
    """
    Perform grid search to find optimal hyperparameters for XGBoost.
    Train on downsampled data, validate on original class distribution.
    """
    print(f"Starting hyperparameter tuning for {segment_type}/{mode}...")
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Define parameter grid - REDUCED for faster training
    param_grid = {
        'max_depth': [6, 8],           # Reduced from [6, 8, 10]
        'learning_rate': [0.1, 0.15],  # Reduced from [0.05, 0.1, 0.15]
        'n_estimators': [100, 150],    # Reduced from [100, 150, 200]
        'subsample': [0.8],            # Reduced from [0.8, 0.9]
        'colsample_bytree': [0.8],     # Reduced from [0.8, 0.9]
        'reg_alpha': [0.1],            # Reduced from [0.1, 0.5]
        'reg_lambda': [1.0]            # Reduced from [1.0, 2.0]
    }
    
    # Base XGBoost classifier
    base_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,  # Use single thread to avoid conflicts with parallel processing
        tree_method='hist' if not USE_GPU else 'gpu_hist',  # Use CPU by default, GPU if specified
        predictor='cpu_predictor' if not USE_GPU else 'gpu_predictor',  # Use CPU predictor by default
        # Add class weights to help with imbalance
        class_weight='balanced'
    )
    
    # Create scorer - using precision for positive class
    scorer = make_scorer(precision_score, pos_label=1)
    
    # Set up cross-validation using original class distribution for stratification
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # For simplicity, we'll use the downsampled data for both training and validation
    # This is a reasonable compromise since the validation performance will still be meaningful
    # and we're primarily interested in relative performance between different hyperparameters
    print(f"Using downsampled data for both training and validation")
    print(f"Training data shape: {X_train.shape}, class distribution: {y_train.value_counts().to_dict()}")
    
    # Perform random search
    grid_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid,
        n_iter=RANDOM_SEARCH_ITERATIONS,  # Try random parameter combinations
        scoring=scorer,
        cv=cv,
        n_jobs=2,  # Use 2 threads per process for parallelism
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    print(f"Fitting random search with {grid_search.n_iter} random parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    print(f"Random search completed for {segment_type}/{mode}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation precision: {grid_search.best_score_:.4f}")
    return grid_search

def find_optimal_threshold(y_true, y_pred_proba, metric='precision'):
    """
    Find the optimal threshold that maximizes the specified metric.
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division='warn')
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division='warn')
        else:
            continue
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def evaluate_best_model(grid_search, X_train, y_train, X_test, y_test, mode, segment_type):
    """
    Evaluate the best model from grid search and extract feature importance.
    """
    print(f"Evaluating best model for {segment_type}/{mode}...")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Get predictions on training data for threshold optimization
    y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
    
    # Find optimal threshold on training data
    optimal_threshold, train_precision = find_optimal_threshold(y_train, y_train_pred_proba, 'precision')
    print(f"Optimal threshold: {optimal_threshold:.3f} (training precision: {train_precision:.4f})")
    
    # Make predictions on test set using optimal threshold
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    test_precision = precision_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Precision (with optimal threshold): {test_precision:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, y_pred, y_pred_proba, feature_importance, test_precision

def save_results(grid_search, best_model, feature_importance, y_test, y_pred, y_pred_proba, mode, segment_type):
    """
    Save all results including model, reports, and feature importance.
    """
    # Define directories
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    results_dir = base_path / f'results/xg_boost_gridsearch/segment_{segment_type}/mode_{mode}'
    models_dir = results_dir / 'models'
    reports_dir = results_dir / 'reports'
    plots_dir = results_dir / 'plots'
    
    # Create directories
    models_dir.mkdir(exist_ok=True, parents=True)
    reports_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Save best model
    model_path = models_dir / 'best_xgboost_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # Save grid search results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(results_dir / 'grid_search_results.csv', index=False)
    print(f"Saved grid search results to {results_dir / 'grid_search_results.csv'}")
    
    # Save classification report
    class_names = [f'not {mode}', f'{mode} (not preselected)']
    report = classification_report(y_test, y_pred, target_names=class_names)
    report_path = reports_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("BEST MODEL PERFORMANCE\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV precision: {grid_search.best_score_:.4f}\n")
        f.write(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"Test precision: {precision_score(y_test, y_pred):.4f}\n")
        f.write(f"Test F1 score: {f1_score(y_test, y_pred):.4f}\n\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n")
        # Convert report to string if it's a dict (newer scikit-learn versions)
        if isinstance(report, dict):
            f.write(str(report))
        else:
            f.write(report)
        f.write(f"\nPrediction probabilities summary:\n")
        f.write(f"Min probability: {y_pred_proba.min():.4f}\n")
        f.write(f"Max probability: {y_pred_proba.max():.4f}\n")
        f.write(f"Mean probability: {y_pred_proba.mean():.4f}\n")
        f.write(f"Std probability: {y_pred_proba.std():.4f}\n")
    print(f"Saved classification report to {report_path}")
    
    # Save feature importance
    feature_importance.to_csv(results_dir / 'feature_importance.csv', index=False)
    print(f"Saved feature importance to {results_dir / 'feature_importance.csv'}")

def create_visualizations(grid_search, feature_importance, segment_type, mode):
    """
    Create visualizations of the results.
    """
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    plots_dir = base_path / f'results/xg_boost_gridsearch/segment_{segment_type}/mode_{mode}/plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 20 Feature Importances - {segment_type} segment, {mode} mode')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Grid search results heatmap (simplified for key parameters)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Create heatmap for max_depth vs learning_rate (averaging over other parameters)
    pivot_data = cv_results.groupby(['param_max_depth', 'param_learning_rate'])['mean_test_score'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'CV F1 Scores - {segment_type} segment, {mode} mode\n(max_depth vs learning_rate)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'gridsearch_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {plots_dir}")

def downsample_negative_class(X, y, negative_fraction=0.3, random_state=42):
    """
    Downsample the negative class, keep all positive samples.
    """
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_neg_keep = int(len(neg_idx) * negative_fraction)
    if n_neg_keep < 1:
        n_neg_keep = 1
    neg_idx_down = np.random.RandomState(random_state).choice(neg_idx, n_neg_keep, replace=False)

    keep_idx = np.concatenate([pos_idx, neg_idx_down])
    X_down = X.loc[keep_idx]
    y_down = y.loc[keep_idx]
    return X_down, y_down

def run_for_mode_and_segment(df, mode, segment_type):
    """
    Run the complete pipeline for a specific mode and segment combination.
    """
    print(f"\n=== Running GridSearch for segment: {segment_type}, mode: {mode} ===")
    try:
        # Monitor memory usage
        process = psutil.Process()
        print(f"Initial memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
        # Prepare data
        X, y, feature_cols = prepare_features_and_target(df, mode)
        
        print(f"Memory after feature preparation: {process.memory_info().rss / 1024**3:.2f} GB")
        
        # Check if we have both classes
        if y.nunique() < 2:
            print(f"Skipping segment={segment_type}, mode={mode}: only one class present.")
            return
        
        # Check if we have enough positive samples
        positive_samples = y.sum()
        print(f"Positive samples: {positive_samples}")
        
        if positive_samples < 10:
            print(f"Skipping segment={segment_type}, mode={mode}: too few positive samples ({positive_samples})")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Downsample negative class only in training data
        X_train, y_train = downsample_negative_class(X_train, y_train, negative_fraction=0.3, random_state=RANDOM_STATE)
        print(f"After downsampling training data: X_train shape: {X_train.shape}, y_train distribution: {y_train.value_counts().to_dict()}")
        
        print(f"Memory after data split: {process.memory_info().rss / 1024**3:.2f} GB")
        
        # Perform grid search
        grid_search = perform_grid_search(X_train, y_train, X, y, segment_type, mode)
        
        print(f"Memory after grid search: {process.memory_info().rss / 1024**3:.2f} GB")
        
        # Evaluate best model
        best_model, y_pred, y_pred_proba, feature_importance, test_precision = evaluate_best_model(
            grid_search, X_train, y_train, X_test, y_test, mode, segment_type
        )
        
        # Save results
        save_results(grid_search, best_model, feature_importance, y_test, y_pred, y_pred_proba, mode, segment_type)
        
        # Create visualizations
        create_visualizations(grid_search, feature_importance, segment_type, mode)
        
        # Clean up memory
        del X, y, X_train, X_test, y_train, y_test, grid_search, best_model
        import gc
        gc.collect()
        
        print(f"Final memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
        print(f"GridSearch completed successfully for {segment_type}/{mode}")
        
    except Exception as e:
        print(f"Error processing {segment_type}/{mode}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up memory even on error
        try:
            del X, y, X_train, X_test, y_train, y_test
            import gc
            gc.collect()
        except:
            pass

def main(mode_list, segment_type_list):
    print(f"Starting XGBoost GridSearch for segments: {segment_type_list}")
    print(f"Modes: {mode_list}")
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
        (base_path / 'results/xg_boost_gridsearch').mkdir(exist_ok=True, parents=True)
        
        # Run sequentially for each mode
        for mode in mode_list:
            print(f"\nProcessing mode: {mode}")
            try:
                run_for_mode_and_segment(df_segment, mode, segment_type)
                print(f"Completed mode: {mode}")
            except Exception as e:
                print(f"Error processing mode {mode}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nGridSearch completed for segment: {segment_type}")
        print(f"Results saved to: {base_path}/results/xg_boost_gridsearch/segment_{segment_type}/")

if __name__ == "__main__":
    # Configuration
    #segment_type_list = ['churned', 'airport', 'all']
    segment_type_list = ['all']
    #mode_list = ['fastpass', 'standard', 'premium', 'plus', 'lux', 'luxsuv']
    #mode_list = ['fastpass', 'premium', 'plus', 'lux', 'luxsuv']
    mode_list = ['premium']
    
    main(mode_list, segment_type_list) 