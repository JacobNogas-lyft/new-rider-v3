import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score
import joblib
from pathlib import Path
from utils.load_data import load_parquet_data
import warnings
warnings.filterwarnings('ignore')

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
    """Filter dataframe by segment type."""
    if segment_type == 'all':
        filtered_df = df.copy()
        print(f"Using all data: {len(filtered_df)} rows")
    else:
        raise ValueError(f"Unknown segment type: {segment_type}")
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
    
    # Categorical columns to drop
    CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                                'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id']
    
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Use all categorical columns except the high cardinality ones
    feature_cols = [col for col in numeric_cols if col not in drop_cols] + [col for col in categorical_cols if (col not in drop_cols) and (col not in CATEGORICAL_COLS_TO_DROP)]
    X = df[feature_cols]
    y = df['target_diff_mode']
    
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Final X shape: {X.shape}")
    print("Data loaded and processed.")
    return X, y, feature_cols

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

def main():
    # Configuration
    segment_type = 'all'
    mode = 'premium'
    model_path = Path('/home/sagemaker-user/studio/src/new-rider-v3/models/xg_boost/all_features/segment_all/mode_premium/max_depth_10/xgboost_model.joblib')
    
    print(f"Loading existing model from: {model_path}")
    
    # Load the trained model
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_parquet_data()
    df = add_churned_indicator(df)
    
    # Filter by segment
    df_segment = filter_by_segment(df, segment_type)
    
    # Filter out rows with missing required columns
    required_cols = ['requested_ride_type', 'preselected_mode']
    df_segment = df_segment.dropna(subset=required_cols)
    
    print(f"Segment data shape: {df_segment.shape}")
    
    # Prepare features and target
    X, y, feature_cols = prepare_features_and_target(df_segment, mode)
    
    # Check if we have both classes
    if y.nunique() < 2:
        print("Only one class present. Exiting.")
        return
    
    # Split data (same as original training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Get predictions on training data
    print("Getting predictions on training data...")
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    optimal_threshold, train_precision = find_optimal_threshold(y_train, y_train_pred_proba, 'precision')
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Training precision with optimal threshold: {train_precision:.4f}")
    
    # Get predictions on test data
    print("Getting predictions on test data...")
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compare default vs optimal threshold
    print("\n" + "="*60)
    print("COMPARISON: Default (0.5) vs Optimal Threshold")
    print("="*60)
    
    # Default threshold (0.5)
    y_test_pred_default = (y_test_pred_proba >= 0.5).astype(int)
    default_precision = precision_score(y_test, y_test_pred_default)
    default_recall = recall_score(y_test, y_test_pred_default)
    default_f1 = f1_score(y_test, y_test_pred_default)
    default_accuracy = accuracy_score(y_test, y_test_pred_default)
    
    # Optimal threshold
    y_test_pred_optimal = (y_test_pred_proba >= optimal_threshold).astype(int)
    optimal_precision = precision_score(y_test, y_test_pred_optimal)
    optimal_recall = recall_score(y_test, y_test_pred_optimal)
    optimal_f1 = f1_score(y_test, y_test_pred_optimal)
    optimal_accuracy = accuracy_score(y_test, y_test_pred_optimal)
    
    print(f"Default threshold (0.5):")
    print(f"  Precision: {default_precision:.4f}")
    print(f"  Recall: {default_recall:.4f}")
    print(f"  F1 Score: {default_f1:.4f}")
    print(f"  Accuracy: {default_accuracy:.4f}")
    
    print(f"\nOptimal threshold ({optimal_threshold:.3f}):")
    print(f"  Precision: {optimal_precision:.4f}")
    print(f"  Recall: {optimal_recall:.4f}")
    print(f"  F1 Score: {optimal_f1:.4f}")
    print(f"  Accuracy: {optimal_accuracy:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Precision: +{optimal_precision - default_precision:.4f} ({((optimal_precision/default_precision - 1)*100):.1f}%)")
    print(f"  Recall: +{optimal_recall - default_recall:.4f} ({((optimal_recall/default_recall - 1)*100):.1f}%)")
    print(f"  F1 Score: +{optimal_f1 - default_f1:.4f} ({((optimal_f1/default_f1 - 1)*100):.1f}%)")
    
    # Class support values
    n_pos = int(y_test.sum())
    n_neg = int((y_test == 0).sum())
    pct_pos = n_pos / (n_pos + n_neg) * 100
    
    # Calculate percentage of positive predictions for each threshold
    pct_pred_pos_default = (y_test_pred_default.sum() / len(y_test_pred_default)) * 100
    pct_pred_pos_optimal = (y_test_pred_optimal.sum() / len(y_test_pred_optimal)) * 100
    
    # Get prediction counts for each threshold
    n_pred_pos_default = int(y_test_pred_default.sum())
    n_pred_neg_default = int((y_test_pred_default == 0).sum())
    n_pred_pos_optimal = int(y_test_pred_optimal.sum())
    n_pred_neg_optimal = int((y_test_pred_optimal == 0).sum())
    
    print(f"\nTest set support:")
    print(f"  Positive class: {n_pos}")
    print(f"  Negative class: {n_neg}")
    print(f"  % positive: {pct_pos:.2f}%")
    
    print(f"\nPrediction distribution:")
    print(f"  Default threshold (0.5): {pct_pred_pos_default:.2f}% predicted as positive")
    print(f"    - Predicted positive: {n_pred_pos_default}")
    print(f"    - Predicted negative: {n_pred_neg_default}")
    print(f"  Optimal threshold (0.850): {pct_pred_pos_optimal:.2f}% predicted as positive")
    print(f"    - Predicted positive: {n_pred_pos_optimal}")
    print(f"    - Predicted negative: {n_pred_neg_optimal}")
    
    # Save results
    results_path = Path('/home/sagemaker-user/studio/src/new-rider-v3/models/xg_boost/all_features/segment_all/mode_premium/max_depth_10/threshold_optimization_results.txt')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("THRESHOLD OPTIMIZATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Optimal threshold: {optimal_threshold:.3f}\n")
        f.write(f"Training precision with optimal threshold: {train_precision:.4f}\n\n")
        f.write("TEST SET PERFORMANCE COMPARISON\n")
        f.write("="*50 + "\n")
        f.write(f"Default threshold (0.5):\n")
        f.write(f"  Precision: {default_precision:.4f}\n")
        f.write(f"  Recall: {default_recall:.4f}\n")
        f.write(f"  F1 Score: {default_f1:.4f}\n")
        f.write(f"  Accuracy: {default_accuracy:.4f}\n\n")
        f.write(f"Optimal threshold ({optimal_threshold:.3f}):\n")
        f.write(f"  Precision: {optimal_precision:.4f}\n")
        f.write(f"  Recall: {optimal_recall:.4f}\n")
        f.write(f"  F1 Score: {optimal_f1:.4f}\n")
        f.write(f"  Accuracy: {optimal_accuracy:.4f}\n\n")
        f.write("IMPROVEMENT\n")
        f.write("="*50 + "\n")
        f.write(f"Precision: +{optimal_precision - default_precision:.4f} ({((optimal_precision/default_precision - 1)*100):.1f}%)\n")
        f.write(f"Recall: +{optimal_recall - default_recall:.4f} ({((optimal_recall/default_recall - 1)*100):.1f}%)\n")
        f.write(f"F1 Score: +{optimal_f1 - default_f1:.4f} ({((optimal_f1/default_f1 - 1)*100):.1f}%)\n")
        f.write("\nTest set support:\n")
        f.write(f"  Positive class: {n_pos}\n")
        f.write(f"  Negative class: {n_neg}\n")
        f.write(f"  % positive: {pct_pos:.2f}%\n")
        f.write("\nPrediction distribution:\n")
        f.write(f"  Default threshold (0.5): {pct_pred_pos_default:.2f}% predicted as positive\n")
        f.write(f"    - Predicted positive: {n_pred_pos_default}\n")
        f.write(f"    - Predicted negative: {n_pred_neg_default}\n")
        f.write(f"  Optimal threshold (0.850): {pct_pred_pos_optimal:.2f}% predicted as positive\n")
        f.write(f"    - Predicted positive: {n_pred_pos_optimal}\n")
        f.write(f"    - Predicted negative: {n_pred_neg_optimal}\n")
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main() 