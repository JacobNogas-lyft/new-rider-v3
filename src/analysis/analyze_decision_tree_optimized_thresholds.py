import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Total sessions from standard/all segment (118634 not standard + 4291 standard not preselected)
TOTAL_SESSIONS = 122925

# --- Helper: Churned user indicator (copied from training files) ---
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
        churned_mask = (df['is_churned_user'] == 1)
        filtered_df = df[churned_mask].copy()
        print(f"Churned rider sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'all':
        filtered_df = df.copy()
        print(f"Using all data: {len(filtered_df)} rows")
    else:
        raise ValueError(f"Unknown segment type: {segment_type}. Use 'airport', 'airport_dropoff', 'airport_pickup', 'churned', or 'all'")
    return filtered_df

def prepare_features_and_target_for_analysis(df, mode):
    """
    EXACT COPY of prepare_features_and_target from training script.
    This ensures we get the exact same features and can use the same train/test split.
    """
    # Use the exact same CATEGORICAL_COLS_TO_DROP as training script
    CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                                'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id',
                                'rider_lyft_id',
                                'signup_at',
                                'destination_place_name',
                                'pickup_place_name']
    
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
    return X, y

def split_data_for_analysis(X, y):
    """
    EXACT COPY of split_data from training script.
    This ensures we get the exact same train/test split as used during training.
    """
    print("Splitting data into train and test sets (stratified)...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_or_create_test_data_for_model(df_segment, df_segment_with_rider_id, segment_type, mode, data_version='v2', force_recreate=False):
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    # Include data_version in cache key to avoid feature mismatch issues
    cache_key = f"test_data_{segment_type}_{mode}_{data_version}"
    cache_file = f'/home/sagemaker-user/studio/src/new-rider-v3/data{data_suffix}/{cache_key}.joblib'
    
    if not force_recreate and os.path.exists(cache_file):
        print(f"Loading cached test data for {cache_key} (data_version={data_version})...")
        try:
            cached_data = joblib.load(cache_file)
            X_test, y_test, rider_ids_test = cached_data
            print(f"✅ Loaded cached test data - Test: {X_test.shape}")
            return X_test, y_test, rider_ids_test
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Will recreate test data...")
    
    print(f"Creating test data for {cache_key} (data_version={data_version})...")
    
    # Use the exact same feature preparation as training script
    X, y = prepare_features_and_target_for_analysis(df_segment, mode)
    
    # Use the exact same train/test split as training script
    X_train, X_test, y_train, y_test = split_data_for_analysis(X, y)
    
    # Extract rider_ids for the test set (for analysis purposes)
    # We need to get the test indices to extract the corresponding rider_ids
    if df_segment_with_rider_id is not None and 'rider_lyft_id' in df_segment_with_rider_id.columns:
        # Get the same test indices by recreating the split on the original dataframe indices
        df_for_split = df_segment_with_rider_id.copy()
        df_for_split['target_diff_mode'] = ((df_for_split['requested_ride_type'] != df_for_split['preselected_mode']) & (df_for_split['requested_ride_type'] == mode)).astype(int)
        
        # Split the indices to match the feature split
        indices = df_for_split.index.values
        _, indices_test, _, _ = train_test_split(
            indices, df_for_split['target_diff_mode'], 
            test_size=0.2, random_state=42, stratify=df_for_split['target_diff_mode']
        )
        rider_ids_test = df_segment_with_rider_id.loc[indices_test, 'rider_lyft_id'].values
    else:
        rider_ids_test = np.array([None] * len(X_test))
    
    cache_dir = f'/home/sagemaker-user/studio/src/new-rider-v3/data{data_suffix}'
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    print(f"Saving test data to cache: {cache_file}")
    joblib.dump((X_test, y_test, rider_ids_test), cache_file)
    print(f"✅ Created and cached test data - Test: {X_test.shape}")
    print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    return X_test, y_test, rider_ids_test

def load_model_and_data(model_path, df_segment=None, df_segment_with_rider_id=None, data_version='v2', force_recreate_cache=False):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        path_parts = model_path.split(os.sep)
        segment = None
        mode = None
        for part in path_parts:
            if part.startswith('segment_'):
                segment = part.replace('segment_', '')
            elif part.startswith('mode_'):
                mode = part.replace('mode_', '')
        if not segment or not mode:
            raise ValueError(f"Could not extract segment and mode from path: {model_path}")
        X_test, y_test, rider_ids_test = load_or_create_test_data_for_model(df_segment, df_segment_with_rider_id, segment, mode, data_version, force_recreate_cache)
        return model, X_test, y_test, rider_ids_test
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None, None

def find_optimal_threshold(model, X_test, y_test, rider_ids_test):
    print("Getting predictions on test data...")
    test_probs = model.predict_proba(X_test)[:, 1]
    print("Finding optimal threshold for maximum F1 score...")
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    best_rider_count = 0
    for threshold in thresholds:
        test_preds = (test_probs >= threshold).astype(int)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        # Count unique rider_lyft_id for positive predictions
        if rider_ids_test is not None:
            unique_riders = len(set(rider_ids_test[test_preds == 1]))
        else:
            unique_riders = 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_rider_count = unique_riders
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Test F1 with optimal threshold: {best_f1:.4f}")
    print(f"Test precision with optimal threshold: {best_precision:.4f}")
    print(f"Test recall with optimal threshold: {best_recall:.4f}")
    print(f"Unique riders with positive prediction (optimal): {best_rider_count}")
    # Default threshold
    default_preds = (test_probs >= 0.5).astype(int)
    default_precision = precision_score(y_test, default_preds, zero_division=0)
    default_recall = recall_score(y_test, default_preds, zero_division=0)
    default_f1 = f1_score(y_test, default_preds, zero_division=0)
    default_accuracy = accuracy_score(y_test, default_preds)
    if rider_ids_test is not None:
        default_rider_count = len(set(rider_ids_test[default_preds == 1]))
    else:
        default_rider_count = 0
    optimal_preds = (test_probs >= best_threshold).astype(int)
    optimal_precision = precision_score(y_test, optimal_preds, zero_division=0)
    optimal_recall = recall_score(y_test, optimal_preds, zero_division=0)
    optimal_f1 = f1_score(y_test, optimal_preds, zero_division=0)
    optimal_accuracy = accuracy_score(y_test, optimal_preds)
    default_positive_preds = np.sum(default_preds)
    optimal_positive_preds = np.sum(optimal_preds)
    total_samples = len(y_test)
    results = {
        'optimal_threshold': best_threshold,
        'default_precision': default_precision,
        'default_recall': default_recall,
        'default_f1': default_f1,
        'default_accuracy': default_accuracy,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'optimal_f1': optimal_f1,
        'optimal_accuracy': optimal_accuracy,
        'default_positive_preds': default_positive_preds,
        'optimal_positive_preds': optimal_positive_preds,
        'total_samples': total_samples,
        'default_positive_ratio': (default_positive_preds / total_samples) * 100,
        'optimal_positive_ratio': (optimal_positive_preds / total_samples) * 100,
        'test_support_positive': np.sum(y_test),
        'test_support_negative': np.sum(y_test == 0),
        'unique_riders_optimal': best_rider_count,
        'unique_riders_default': default_rider_count
    }
    return results

def compare_model_features(segment, mode, max_depth=10):
    """Compare expected features between v2 and v3 models for the same segment/mode."""
    print(f"\n{'='*80}")
    print(f"COMPARING MODEL FEATURES: {segment}/{mode} (max_depth={max_depth})")
    print(f"{'='*80}")
    
    results = {}
    
    for data_version in ['v2', 'v3']:
        data_suffix = f"_{data_version}" if data_version != 'original' else ""
        model_path = f'/home/sagemaker-user/studio/src/new-rider-v3/models/decision_tree/all_features{data_suffix}/segment_{segment}/mode_{mode}/max_depth_{max_depth}/decision_tree_model.joblib'
        
        print(f"\n--- {data_version.upper()} Model ---")
        print(f"Path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found for {data_version}")
            results[data_version] = None
            continue
            
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully")
            
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
                print(f"📊 Model expects {len(features)} features")
                results[data_version] = set(features)
            elif hasattr(model, 'n_features_in_'):
                print(f"📊 Model expects {model.n_features_in_} features (no feature names available)")
                results[data_version] = f"n_features_in_: {model.n_features_in_}"
            else:
                print(f"❌ Cannot determine feature information")
                results[data_version] = None
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            results[data_version] = None
    
    # Compare features if both models loaded successfully
    if results['v2'] is not None and results['v3'] is not None:
        if isinstance(results['v2'], set) and isinstance(results['v3'], set):
            print(f"\n--- FEATURE COMPARISON ---")
            
            v2_features = results['v2']
            v3_features = results['v3']
            
            print(f"V2 features: {len(v2_features)}")
            print(f"V3 features: {len(v3_features)}")
            
            # Features only in v2
            v2_only = v2_features - v3_features
            if v2_only:
                print(f"\n🔴 Features only in V2 ({len(v2_only)}):")
                for i, feature in enumerate(sorted(v2_only)[:10]):  # Show first 10
                    print(f"  {i+1}. {feature}")
                if len(v2_only) > 10:
                    print(f"  ... and {len(v2_only) - 10} more")
            
            # Features only in v3
            v3_only = v3_features - v2_features
            if v3_only:
                print(f"\n🟢 Features only in V3 ({len(v3_only)}):")
                for i, feature in enumerate(sorted(v3_only)[:10]):  # Show first 10
                    print(f"  {i+1}. {feature}")
                if len(v3_only) > 10:
                    print(f"  ... and {len(v3_only) - 10} more")
            
            # Common features
            common = v2_features & v3_features
            print(f"\n✅ Common features: {len(common)}")
            
            if len(v2_only) == 0 and len(v3_only) == 0:
                print("🎉 Perfect match! All features are identical.")
            else:
                print(f"⚠️  Feature mismatch detected!")
                
        else:
            print(f"\n--- COMPARISON RESULTS ---")
            print(f"V2: {results['v2']}")
            print(f"V3: {results['v3']}")
    
    return results

def clear_old_cache_files(data_version='v2'):
    """Clear old cache files that might have incompatible features."""
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    cache_dir = f'/home/sagemaker-user/studio/src/new-rider-v3/data{data_suffix}'
    
    if not os.path.exists(cache_dir):
        return
    
    # Clear old cache files that don't have data_version in the key
    old_pattern = re.compile(r'test_data_\w+_\w+\.joblib$')  # Without data_version
    new_pattern = re.compile(r'test_data_\w+_\w+_v\d+\.joblib$')  # With data_version
    
    cleared_count = 0
    for filename in os.listdir(cache_dir):
        if old_pattern.match(filename) and not new_pattern.match(filename):
            file_path = os.path.join(cache_dir, filename)
            try:
                os.remove(file_path)
                print(f"Cleared old cache file: {filename}")
                cleared_count += 1
            except Exception as e:
                print(f"Error clearing {filename}: {e}")
    
    if cleared_count > 0:
        print(f"✅ Cleared {cleared_count} old cache files")
    else:
        print("No old cache files to clear")

def analyze_all_models_with_optimization(data_version='v2', force_recreate_cache=False):
    all_results = []
    
    # Clear old cache files to avoid feature mismatch issues
    print("Clearing old cache files...")
    clear_old_cache_files(data_version)
    
    # Determine the base path based on data_version
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    base_models_path = f'/home/sagemaker-user/studio/src/new-rider-v3/models/decision_tree/all_features{data_suffix}'
    
    print(f"Analyzing models from: {base_models_path}")
    
    # Check if directory exists
    if not os.path.exists(base_models_path):
        print(f"ERROR: Directory {base_models_path} does not exist!")
        return pd.DataFrame(all_results)
    
    # Load data once at the beginning (EXACT COPY of training script preprocessing)
    print("Loading data once for all analysis...")
    from utils.load_data import load_parquet_data
    df_original = load_parquet_data(data_version)
    df_original = add_churned_indicator(df_original)
    
    # Keep a copy with rider_lyft_id for analysis purposes
    df_with_rider_id = df_original.copy()
    
    # Apply the EXACT same preprocessing as training script
    if data_version in ['v2', 'v3']:
        assert 'rider_lyft_id' in df_original.columns, f"rider_lyft_id should be in columns when data_version={data_version}, but not found. Available columns: {[col for col in df_original.columns if 'session' in col]}"
        print(f"Verified: rider_lyft_id is in columns for {data_version.upper()} data")

    df = df_original.drop_duplicates(subset=['purchase_session_id'], keep='first')
    print(f"After deduplication: {len(df)} rows")
    
    # Assert that we have exactly 1 row per purchase_session_id
    assert df['purchase_session_id'].nunique() == len(df), f"Expected 1 row per purchase_session_id, but got {len(df)} rows for {df['purchase_session_id'].nunique()} unique purchase_session_ids"
    print(f"Verified: {len(df)} rows with {df['purchase_session_id'].nunique()} unique purchase_session_ids")

    # Use the EXACT same CATEGORICAL_COLS_TO_DROP as training script
    CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                                'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id',
                                'rider_lyft_id',
                                'signup_at',
                                'destination_place_name',
                                'pickup_place_name']
    df.drop(columns=CATEGORICAL_COLS_TO_DROP, inplace=True)
    
    # Assert that rider_lyft_id is not in columns (matches training script)
    assert 'rider_lyft_id' not in df.columns, f"rider_lyft_id should not be in columns, but found: {[col for col in df.columns if 'rider_lyft_id' in col]}"
    print(f"Verified: rider_lyft_id not in columns")
    
    # Keep the version with rider_lyft_id for segment processing
    df_with_rider_id = df_with_rider_id.drop_duplicates(subset=['purchase_session_id'], keep='first')
    print(f"Keeping parallel dataframe with rider_lyft_id for analysis purposes")
    
    # Group models by segment for efficient processing
    segment_models = {}
    
    for root, _, files in os.walk(base_models_path):
        for file in files:
            if file != 'decision_tree_model.joblib':
                continue
            file_path = os.path.join(root, file)
            path_parts = file_path.split(os.sep)
            
            # Only include models in all_features directory (with or without _v2 suffix)
            all_features_found = False
            for part in path_parts:
                if 'all_features' in part:
                    all_features_found = True
                    break
            
            if not all_features_found:
                continue
            
            # Include all max_depth values: 3, 5, 10
            max_depth_found = False
            for part in path_parts:
                if part.startswith('max_depth_'):
                    depth_value = part.replace('max_depth_', '')
                    if depth_value in ['3', '5', '10']:
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
            
            if segment not in segment_models:
                segment_models[segment] = []
            segment_models[segment].append((file_path, mode, int(depth)))
    
    # --- Store global unique riders from 'all' segment test set ---
    global_rider_set = set()
    
    # Process each segment
    for segment_type, models in segment_models.items():
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_type}")
        print(f"{'='*60}")
        
        # Filter data for this segment (following training script pattern)
        df_segment = filter_by_segment(df, segment_type)
        df_segment_with_rider_id = filter_by_segment(df_with_rider_id, segment_type)
        
        # Filter out rows with missing required columns
        required_cols = ['requested_ride_type', 'preselected_mode']
        df_segment = df_segment.dropna(subset=required_cols)
        df_segment_with_rider_id = df_segment_with_rider_id.dropna(subset=required_cols)
        print(f"Segment data shape: {df_segment.shape}")
        
        # Process each model for this segment
        for i, (file_path, mode, depth) in enumerate(models):
            print(f"\n{'='*60}")
            print(f"Analyzing: {segment_type}/{mode} (max_depth={depth})")
            print(f"{'='*60}")
            
            try:
                model, X_test, y_test, rider_ids_test = load_model_and_data(file_path, df_segment, df_segment_with_rider_id, data_version, force_recreate_cache)
                if model is None:
                    print(f"Skipping {file_path} due to loading error")
                    continue
                
                # For the 'all' segment, save the global set of unique riders
                if segment_type == 'all' and i == 0 and rider_ids_test is not None:
                    global_rider_set = set(rider_ids_test)
                    print(f"Global unique riders in 'all' segment test set: {len(global_rider_set)}")
                
                results = find_optimal_threshold(model, X_test, y_test, rider_ids_test)
                results.update({
                    'segment': segment_type,
                    'mode': mode,
                    'max_depth': depth,
                    'model_path': file_path,
                    'data_version': data_version.upper(),
                    'rider_ids_test': rider_ids_test  # Save for later
                })
                all_results.append(results)
                print(f"✅ Completed analysis for {segment_type}/{mode}")
            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {e}")
                continue
    
    # After all segments processed, use global_rider_set for all percentage calculations
    if len(all_results) > 0:
        df_results = pd.DataFrame(all_results)
        global_rider_count = len(global_rider_set)
        print(f"Global denominator for all heatmap cells: {global_rider_count}")
        df_results['total_unique_riders'] = global_rider_count
        df_results['pct_riders_positive_optimal'] = df_results['unique_riders_optimal'] / global_rider_count
        df_results['pct_riders_positive_default'] = df_results['unique_riders_default'] / global_rider_count
        return df_results
    else:
        return pd.DataFrame(all_results)

def create_optimized_analysis_plots(df, data_version='v2'):
    """Create visualization plots for the optimized threshold analysis."""
    plt.style.use('default')
    sns.set_palette("YlGn")
    vmin, vmax = 0, 1
    
    # Add data version to title
    data_version_title = data_version.upper()
    
    fig, axes = plt.subplots(5, 3, figsize=(24, 30))
    fig.suptitle(f'Decision Tree Model Performance: Optimized vs Default Thresholds - {data_version_title} Data', fontsize=16, fontweight='bold')
    
    # Row 1: Default Threshold Performance
    ax1 = axes[0, 0]
    pivot_default_precision = df.pivot_table(index='mode', columns='segment', values='default_precision', aggfunc='mean')
    sns.heatmap(pivot_default_precision, annot=True, fmt='.3f', cmap='YlGn', ax=ax1, center=0.5, vmin=vmin, vmax=vmax)
    ax1.set_title('Default Threshold (0.5) - Precision')
    ax2 = axes[0, 1]
    pivot_default_recall = df.pivot_table(index='mode', columns='segment', values='default_recall', aggfunc='mean')
    sns.heatmap(pivot_default_recall, annot=True, fmt='.3f', cmap='YlGn', ax=ax2, center=0.5, vmin=vmin, vmax=vmax)
    ax2.set_title('Default Threshold (0.5) - Recall')
    ax3 = axes[0, 2]
    pivot_default_f1 = df.pivot_table(index='mode', columns='segment', values='default_f1', aggfunc='mean')
    sns.heatmap(pivot_default_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax3, center=0.5, vmin=vmin, vmax=vmax)
    ax3.set_title('Default Threshold (0.5) - F1 Score')
    
    # Row 2: Optimized Threshold Performance
    ax4 = axes[1, 0]
    pivot_optimal_precision = df.pivot_table(index='mode', columns='segment', values='optimal_precision', aggfunc='mean')
    sns.heatmap(pivot_optimal_precision, annot=True, fmt='.3f', cmap='YlGn', ax=ax4, center=0.5, vmin=vmin, vmax=vmax)
    ax4.set_title('Optimized Threshold - Precision')
    ax5 = axes[1, 1]
    pivot_optimal_recall = df.pivot_table(index='mode', columns='segment', values='optimal_recall', aggfunc='mean')
    sns.heatmap(pivot_optimal_recall, annot=True, fmt='.3f', cmap='YlGn', ax=ax5, center=0.5, vmin=vmin, vmax=vmax)
    ax5.set_title('Optimized Threshold - Recall')
    ax6 = axes[1, 2]
    pivot_optimal_f1 = df.pivot_table(index='mode', columns='segment', values='optimal_f1', aggfunc='mean')
    sns.heatmap(pivot_optimal_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax6, center=0.5, vmin=vmin, vmax=vmax)
    ax6.set_title('Optimized Threshold - F1 Score')
    
    # Row 3: Improvements and Analysis
    ax7 = axes[2, 0]
    df['precision_improvement'] = df['optimal_precision'] - df['default_precision']
    pivot_precision_improvement = df.pivot_table(index='mode', columns='segment', values='precision_improvement', aggfunc='mean')
    sns.heatmap(pivot_precision_improvement, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7, center=0)
    ax7.set_title('Precision Improvement (Optimal - Default)')
    ax8 = axes[2, 1]
    df['f1_improvement'] = df['optimal_f1'] - df['default_f1']
    pivot_f1_improvement = df.pivot_table(index='mode', columns='segment', values='f1_improvement', aggfunc='mean')
    sns.heatmap(pivot_f1_improvement, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax8, center=0)
    ax8.set_title('F1 Score Improvement (Optimal - Default)')
    ax9 = axes[2, 2]
    pivot_optimal_threshold = df.pivot_table(index='mode', columns='segment', values='optimal_threshold', aggfunc='mean')
    sns.heatmap(pivot_optimal_threshold, annot=True, fmt='.3f', cmap='YlGn', ax=ax9, vmin=0.3, vmax=0.9)
    ax9.set_title('Optimal Threshold Values')
    
    # Row 4: Positive Prediction Ratios (relative to total samples)
    total_samples = 122925
    ax10 = axes[3, 0]
    pivot_default_pos_count = df.pivot_table(index='mode', columns='segment', values='default_positive_preds', aggfunc='mean')
    pivot_default_pos_ratio_global = (pivot_default_pos_count / total_samples) * 100
    # Handle NaN values in annotation
    annot_default = pivot_default_pos_ratio_global.round(2).fillna(0).astype(str) + '% (' + pivot_default_pos_count.round(0).fillna(0).astype(int).astype(str) + ')'
    sns.heatmap(pivot_default_pos_ratio_global, annot=annot_default, fmt='', cmap='YlGn', ax=ax10, vmin=0, vmax=100)
    ax10.set_title('Default Threshold - % Predicted Positive (Global)')
    ax11 = axes[3, 1]
    pivot_optimal_pos_count = df.pivot_table(index='mode', columns='segment', values='optimal_positive_preds', aggfunc='mean')
    pivot_optimal_pos_ratio_global = (pivot_optimal_pos_count / total_samples) * 100
    # Handle NaN values in annotation
    annot_optimal = pivot_optimal_pos_ratio_global.round(2).fillna(0).astype(str) + '% (' + pivot_optimal_pos_count.round(0).fillna(0).astype(int).astype(str) + ')'
    sns.heatmap(pivot_optimal_pos_ratio_global, annot=annot_optimal, fmt='', cmap='YlGn', ax=ax11, vmin=0, vmax=100)
    ax11.set_title('Optimized Threshold - % Predicted Positive (Global)')
    
    # 5. Percentage of riders receiving a positive prediction (default threshold)
    ax12 = axes[3, 2]
    pct_riders_default = df.pivot_table(index='mode', columns='segment', values='pct_riders_positive_default', aggfunc='mean')
    sns.heatmap(pct_riders_default, annot=True, fmt='.2%', cmap='YlGn', ax=ax12, vmin=0, vmax=1)
    ax12.set_title('% Riders w/ Positive Prediction (Default)')
    
    # 6. Percentage of riders receiving a positive prediction (optimal threshold)
    ax13 = axes[4, 0]
    pct_riders_optimal = df.pivot_table(index='mode', columns='segment', values='pct_riders_positive_optimal', aggfunc='mean')
    sns.heatmap(pct_riders_optimal, annot=True, fmt='.2%', cmap='YlGn', ax=ax13, vmin=0, vmax=1)
    ax13.set_title('% Riders w/ Positive Prediction (Optimal)')
    
    # Hide unused subplots
    ax14 = axes[4, 1]
    ax14.axis('off')
    ax15 = axes[4, 2]
    ax15.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save with data version suffix
    data_suffix = f"_{data_version}" if data_version != 'original' else ""
    plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimized_analysis_plots_by_depth(df, data_version='v2'):
    """Create separate visualization plots for each max_depth value."""
    plt.style.use('default')
    sns.set_palette("YlGn")
    vmin, vmax = 0, 1
    
    # Add data version to title
    data_version_title = data_version.upper()
    
    # Get unique max_depth values
    max_depths = sorted(df['max_depth'].unique())
    
    for max_depth in max_depths:
        print(f"Creating plots for max_depth={max_depth}...")
        
        # Filter data for this max_depth
        df_depth = df[df['max_depth'] == max_depth].copy()
        
        if len(df_depth) == 0:
            print(f"No data found for max_depth={max_depth}, skipping...")
            continue
        
        fig, axes = plt.subplots(5, 3, figsize=(24, 30))
        fig.suptitle(f'Decision Tree Model Performance: Optimized vs Default Thresholds - {data_version_title} Data (max_depth={max_depth})', fontsize=16, fontweight='bold')
        
        # Row 1: Default Threshold Performance
        ax1 = axes[0, 0]
        pivot_default_precision = df_depth.pivot_table(index='mode', columns='segment', values='default_precision', aggfunc='mean')
        sns.heatmap(pivot_default_precision, annot=True, fmt='.3f', cmap='YlGn', ax=ax1, center=0.5, vmin=vmin, vmax=vmax)
        ax1.set_title('Default Threshold (0.5) - Precision')
        ax2 = axes[0, 1]
        pivot_default_recall = df_depth.pivot_table(index='mode', columns='segment', values='default_recall', aggfunc='mean')
        sns.heatmap(pivot_default_recall, annot=True, fmt='.3f', cmap='YlGn', ax=ax2, center=0.5, vmin=vmin, vmax=vmax)
        ax2.set_title('Default Threshold (0.5) - Recall')
        ax3 = axes[0, 2]
        pivot_default_f1 = df_depth.pivot_table(index='mode', columns='segment', values='default_f1', aggfunc='mean')
        sns.heatmap(pivot_default_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax3, center=0.5, vmin=vmin, vmax=vmax)
        ax3.set_title('Default Threshold (0.5) - F1 Score')
        
        # Row 2: Optimized Threshold Performance
        ax4 = axes[1, 0]
        pivot_optimal_precision = df_depth.pivot_table(index='mode', columns='segment', values='optimal_precision', aggfunc='mean')
        sns.heatmap(pivot_optimal_precision, annot=True, fmt='.3f', cmap='YlGn', ax=ax4, center=0.5, vmin=vmin, vmax=vmax)
        ax4.set_title('Optimized Threshold - Precision')
        ax5 = axes[1, 1]
        pivot_optimal_recall = df_depth.pivot_table(index='mode', columns='segment', values='optimal_recall', aggfunc='mean')
        sns.heatmap(pivot_optimal_recall, annot=True, fmt='.3f', cmap='YlGn', ax=ax5, center=0.5, vmin=vmin, vmax=vmax)
        ax5.set_title('Optimized Threshold - Recall')
        ax6 = axes[1, 2]
        pivot_optimal_f1 = df_depth.pivot_table(index='mode', columns='segment', values='optimal_f1', aggfunc='mean')
        sns.heatmap(pivot_optimal_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax6, center=0.5, vmin=vmin, vmax=vmax)
        ax6.set_title('Optimized Threshold - F1 Score')
        
        # Row 3: Improvements and Analysis
        ax7 = axes[2, 0]
        df_depth['precision_improvement'] = df_depth['optimal_precision'] - df_depth['default_precision']
        pivot_precision_improvement = df_depth.pivot_table(index='mode', columns='segment', values='precision_improvement', aggfunc='mean')
        sns.heatmap(pivot_precision_improvement, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7, center=0)
        ax7.set_title('Precision Improvement (Optimal - Default)')
        ax8 = axes[2, 1]
        df_depth['f1_improvement'] = df_depth['optimal_f1'] - df_depth['default_f1']
        pivot_f1_improvement = df_depth.pivot_table(index='mode', columns='segment', values='f1_improvement', aggfunc='mean')
        sns.heatmap(pivot_f1_improvement, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax8, center=0)
        ax8.set_title('F1 Score Improvement (Optimal - Default)')
        ax9 = axes[2, 2]
        pivot_optimal_threshold = df_depth.pivot_table(index='mode', columns='segment', values='optimal_threshold', aggfunc='mean')
        sns.heatmap(pivot_optimal_threshold, annot=True, fmt='.3f', cmap='YlGn', ax=ax9, vmin=0.3, vmax=0.9)
        ax9.set_title('Optimal Threshold Values')
        
        # Row 4: Positive Prediction Ratios (relative to total samples)
        total_samples = 122925
        ax10 = axes[3, 0]
        pivot_default_pos_count = df_depth.pivot_table(index='mode', columns='segment', values='default_positive_preds', aggfunc='mean')
        pivot_default_pos_ratio_global = (pivot_default_pos_count / total_samples) * 100
        # Handle NaN values in annotation
        annot_default = pivot_default_pos_ratio_global.round(2).fillna(0).astype(str) + '% (' + pivot_default_pos_count.round(0).fillna(0).astype(int).astype(str) + ')'
        sns.heatmap(pivot_default_pos_ratio_global, annot=annot_default, fmt='', cmap='YlGn', ax=ax10, vmin=0, vmax=100)
        ax10.set_title('Default Threshold - % Predicted Positive (Global)')
        ax11 = axes[3, 1]
        pivot_optimal_pos_count = df_depth.pivot_table(index='mode', columns='segment', values='optimal_positive_preds', aggfunc='mean')
        pivot_optimal_pos_ratio_global = (pivot_optimal_pos_count / total_samples) * 100
        # Handle NaN values in annotation
        annot_optimal = pivot_optimal_pos_ratio_global.round(2).fillna(0).astype(str) + '% (' + pivot_optimal_pos_count.round(0).fillna(0).astype(int).astype(str) + ')'
        sns.heatmap(pivot_optimal_pos_ratio_global, annot=annot_optimal, fmt='', cmap='YlGn', ax=ax11, vmin=0, vmax=100)
        ax11.set_title('Optimized Threshold - % Predicted Positive (Global)')
        
        # 5. Percentage of riders receiving a positive prediction (default threshold)
        ax12 = axes[3, 2]
        pct_riders_default = df_depth.pivot_table(index='mode', columns='segment', values='pct_riders_positive_default', aggfunc='mean')
        sns.heatmap(pct_riders_default, annot=True, fmt='.2%', cmap='YlGn', ax=ax12, vmin=0, vmax=1)
        ax12.set_title('% Riders w/ Positive Prediction (Default)')
        
        # 6. Percentage of riders receiving a positive prediction (optimal threshold)
        ax13 = axes[4, 0]
        pct_riders_optimal = df_depth.pivot_table(index='mode', columns='segment', values='pct_riders_positive_optimal', aggfunc='mean')
        sns.heatmap(pct_riders_optimal, annot=True, fmt='.2%', cmap='YlGn', ax=ax13, vmin=0, vmax=1)
        ax13.set_title('% Riders w/ Positive Prediction (Optimal)')
        
        # Hide unused subplots
        ax14 = axes[4, 1]
        ax14.axis('off')
        ax15 = axes[4, 2]
        ax15.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save with data version and max_depth suffix
        data_suffix = f"_{data_version}" if data_version != 'original' else ""
        plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}_max_depth_{max_depth}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}_max_depth_{max_depth}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved for max_depth={max_depth}")

def print_optimized_analysis(df, data_version='v2'):
    """Print detailed analysis of the optimized threshold results."""
    data_version_title = data_version.upper()
    
    print("=" * 80)
    print(f"DECISION TREE OPTIMIZED THRESHOLD ANALYSIS - {data_version_title} DATA")
    print("=" * 80)
    print(f"\nTotal models analyzed: {len(df)}")
    print(f"Data version: {data_version_title}")
    print(f"Segments: {df['segment'].unique()}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Max depths: {df['max_depth'].unique()}")
    expected_total = 122925
    actual_total = df['total_samples'].iloc[0] if len(df) > 0 else 0
    print(f"\n📊 SANITY CHECK:")
    print(f"  Expected total samples: {expected_total}")
    print(f"  Actual total samples: {actual_total}")
    print(f"  Match: {'✅' if actual_total == expected_total else '❌'}")
    print("\n" + "=" * 50)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"\nDefault Threshold (0.5) Average Metrics:")
    print(f"  Precision: {df['default_precision'].mean():.3f}")
    print(f"  Recall: {df['default_recall'].mean():.3f}")
    print(f"  F1 Score: {df['default_f1'].mean():.3f}")
    print(f"  Accuracy: {df['default_accuracy'].mean():.3f}")
    print(f"  Prediction Ratio: {df['default_positive_ratio'].mean():.1f}%")
    print(f"\nOptimized Threshold Average Metrics:")
    print(f"  Precision: {df['optimal_precision'].mean():.3f}")
    print(f"  Recall: {df['optimal_recall'].mean():.3f}")
    print(f"  F1 Score: {df['optimal_f1'].mean():.3f}")
    print(f"  Accuracy: {df['optimal_accuracy'].mean():.3f}")
    print(f"  Prediction Ratio: {df['optimal_positive_ratio'].mean():.1f}%")
    print(f"\nAverage Improvements:")
    print(f"  Precision: +{df['optimal_precision'].mean() - df['default_precision'].mean():.3f}")
    print(f"  Recall: {df['optimal_recall'].mean() - df['default_recall'].mean():+.3f}")
    print(f"  F1 Score: +{df['optimal_f1'].mean() - df['default_f1'].mean():.3f}")
    print(f"  Accuracy: +{df['optimal_accuracy'].mean() - df['default_accuracy'].mean():.3f}")
    print(f"  Prediction Ratio: {df['optimal_positive_ratio'].mean() - df['default_positive_ratio'].mean():+.1f}%")
    print(f"\nAverage Optimal Threshold: {df['optimal_threshold'].mean():.3f}")
    print(f"Threshold Range: {df['optimal_threshold'].min():.3f} - {df['optimal_threshold'].max():.3f}")
    print("\n" + "=" * 50)
    print("BEST PERFORMING MODELS (Optimized Threshold)")
    print("=" * 50)
    df_sorted = df.sort_values('optimal_f1', ascending=False)
    print(f"\n🏆 TOP 5 PERFORMERS (by Optimized F1 Score):")
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"  {i}. {row['segment']}/{row['mode']} (max_depth={row['max_depth']})")
        print(f"     - F1: {row['optimal_f1']:.3f} (threshold: {row['optimal_threshold']:.3f})")
        print(f"     - Precision: {row['optimal_precision']:.3f}")
        print(f"     - Recall: {row['optimal_recall']:.3f}")
        print(f"     - Prediction Ratio: {row['optimal_positive_ratio']:.1f}%")
        print(f"     - Default F1: {row['default_f1']:.3f}")
        print(f"     - F1 Improvement: +{row['optimal_f1'] - row['default_f1']:.3f}")
    print("\n" + "=" * 50)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 50)
    print(f"\nDefault Threshold Prediction Distribution:")
    print(f"  Average % predicted positive: {df['default_positive_ratio'].mean():.1f}%")
    print(f"  Range: {df['default_positive_ratio'].min():.1f}% - {df['default_positive_ratio'].max():.1f}%")
    print(f"\nOptimized Threshold Prediction Distribution:")
    print(f"  Average % predicted positive: {df['optimal_positive_ratio'].mean():.1f}%")
    print(f"  Range: {df['optimal_positive_ratio'].min():.1f}% - {df['optimal_positive_ratio'].max():.1f}%")
    print(f"\nReduction in Positive Predictions:")
    reduction = df['default_positive_ratio'] - df['optimal_positive_ratio']
    print(f"  Average reduction: {reduction.mean():.1f}%")
    print(f"  Range: {reduction.min():.1f}% - {reduction.max():.1f}%")
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    best_overall = df.loc[df['optimal_f1'].idxmax()]
    print(f"\n🏆 BEST OVERALL PERFORMER:")
    print(f"  {best_overall['segment']}/{best_overall['mode']}")
    print(f"  - Optimal F1: {best_overall['optimal_f1']:.3f}")
    print(f"  - Optimal Threshold: {best_overall['optimal_threshold']:.3f}")
    print(f"  - Precision: {best_overall['optimal_precision']:.3f}")
    print(f"  - Recall: {best_overall['optimal_recall']:.3f}")
    print(f"  - Prediction Ratio: {best_overall['optimal_positive_ratio']:.1f}%")
    high_precision = df[df['optimal_precision'] > 0.7]
    if len(high_precision) > 0:
        print(f"\n🎯 MODELS WITH HIGH PRECISION (>0.7) AFTER OPTIMIZATION:")
        for _, row in high_precision.iterrows():
            print(f"  - {row['segment']}/{row['mode']}: {row['optimal_precision']:.3f} precision ({row['optimal_positive_ratio']:.1f}% predicted positive)")
    df['f1_improvement'] = df['optimal_f1'] - df['default_f1']
    significant_improvement = df[df['f1_improvement'] > 0.1]
    if len(significant_improvement) > 0:
        print(f"\n📈 MODELS WITH SIGNIFICANT F1 IMPROVEMENT (>0.1):")
        for _, row in significant_improvement.iterrows():
            print(f"  - {row['segment']}/{row['mode']}: +{row['f1_improvement']:.3f} F1 improvement")
    print(f"\n💡 KEY INSIGHTS:")
    if df['optimal_precision'].mean() > df['default_precision'].mean() + 0.1:
        print("  - Threshold optimization significantly improves precision across models")
    if df['optimal_f1'].mean() > df['default_f1'].mean() + 0.05:
        print("  - Threshold optimization improves overall F1 score")
    if df['optimal_threshold'].mean() > 0.7:
        print("  - Most models benefit from higher thresholds (>0.7) for better precision")
    print("  - Consider implementing threshold optimization in production models")

if __name__ == "__main__":
    # Set data version: 'original', 'v2', or 'v3'
    data_version = 'v3'
    
    # OPTION 1: Run feature comparison (uncomment to compare features between v2 and v3)
    # print("=" * 80)
    # print("FEATURE COMPARISON MODE")
    # print("=" * 80)
    # compare_model_features('all', 'standard', max_depth=10)
    # compare_model_features('airport', 'luxsuv', max_depth=10)
    # exit()
    
    # OPTION 2: Run full analysis (default)
    Path('/home/sagemaker-user/studio/src/new-rider-v3/plots').mkdir(exist_ok=True, parents=True)
    Path('/home/sagemaker-user/studio/src/new-rider-v3/reports').mkdir(exist_ok=True, parents=True)
    print("Starting Decision Tree model analysis with threshold optimization...")
    # Force cache recreation for v3 since it has new features
    df = analyze_all_models_with_optimization(data_version, force_recreate_cache=True)
    if len(df) > 0:
        print_optimized_analysis(df, data_version)
        create_optimized_analysis_plots(df, data_version)
        create_optimized_analysis_plots_by_depth(df, data_version)
        
        # Save the analysis to CSV with data version suffix
        data_suffix = f"_{data_version}" if data_version != 'original' else ""
        csv_path = f'/home/sagemaker-user/studio/src/new-rider-v3/reports/optimized_threshold_analysis_summary_decision_tree{data_suffix}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Analysis saved to '{csv_path}'")
        
        plot_suffix = f"_{data_version}" if data_version != 'original' else ""
        print(f"📈 Plots saved to '/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{plot_suffix}.pdf' and '.png'")
        print(f"📈 Separate plots created for each max_depth value: optimized_threshold_analysis_summary_decision_tree{plot_suffix}_max_depth_[3,5,10].pdf and .png")
    else:
        print(f"No Decision Tree models found to analyze for {data_version} data.") 