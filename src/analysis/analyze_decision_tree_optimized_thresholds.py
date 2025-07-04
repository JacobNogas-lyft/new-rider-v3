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
    ... (same as in XGBoost script)
    """
    if segment_type == 'airport':
        airport_mask = (
            (df['destination_venue_category'] == 'airport') | 
            (df['origin_venue_category'] == 'airport')
        )
        filtered_df = df[airport_mask].copy()
        print(f"Airport sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'churned':
        churned_mask = (df['is_churned_user'] == 1)
        filtered_df = df[churned_mask].copy()
        print(f"Churned rider sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'all':
        filtered_df = df.copy()
        print(f"Using all data: {len(filtered_df)} rows")
    else:
        raise ValueError(f"Unknown segment type: {segment_type}. Use 'airport', 'churned', or 'all'")
    return filtered_df

def prepare_features_target_and_rider_id(df, mode):
    df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == mode)).astype(int)
    print("Creating percentage features for lifetime rides...")
    df['percent_rides_standard_lifetime'] = (
        df['rides_standard_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    df['percent_rides_premium_lifetime'] = (
        df['rides_premium_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    df['percent_rides_plus_lifetime'] = (
        df['rides_plus_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                                'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id']
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Ensure rider_lyft_id is not included in features
    feature_cols = [col for col in numeric_cols if col not in drop_cols and col != 'rider_lyft_id'] + [col for col in categorical_cols if (col not in drop_cols) and (col not in CATEGORICAL_COLS_TO_DROP) and col != 'rider_lyft_id']
    X = df[feature_cols]
    y = df['target_diff_mode']
    # Keep rider_lyft_id for later analysis
    rider_ids = df['rider_lyft_id'].values if 'rider_lyft_id' in df.columns else np.array([None]*len(df))
    print(f"Before get_dummies - X shape: {X.shape}")
    X = pd.get_dummies(X, drop_first=True)
    print(f"After get_dummies - X shape: {X.shape}")
    print("Data loaded and processed.")
    return X, y, rider_ids

def load_or_create_test_data_for_model(df_segment, segment_type, mode, use_v2=False):
    data_suffix = "_v2" if use_v2 else ""
    cache_key = f"test_data_{segment_type}_{mode}"
    cache_file = f'/home/sagemaker-user/studio/src/new-rider-v3/data{data_suffix}/{cache_key}.joblib'
    if os.path.exists(cache_file):
        print(f"Loading cached test data for {cache_key} (V2={use_v2})...")
        try:
            cached_data = joblib.load(cache_file)
            X_test, y_test, rider_ids_test = cached_data
            print(f"✅ Loaded cached test data - Test: {X_test.shape}")
            return X_test, y_test, rider_ids_test
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Will recreate test data...")
    print(f"Creating test data for {cache_key} (V2={use_v2})...")
    X, y, rider_ids = prepare_features_target_and_rider_id(df_segment, mode)
    X_train, X_test, y_train, y_test, rider_ids_train, rider_ids_test = train_test_split(
        X, y, rider_ids, test_size=0.2, random_state=42, stratify=y
    )
    cache_dir = f'/home/sagemaker-user/studio/src/new-rider-v3/data{data_suffix}'
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    print(f"Saving test data to cache: {cache_file}")
    joblib.dump((X_test, y_test, rider_ids_test), cache_file)
    print(f"✅ Created and cached test data - Test: {X_test.shape}")
    print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    return X_test, y_test, rider_ids_test

def load_model_and_data(model_path, df_segment=None, use_v2=False):
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
        X_test, y_test, rider_ids_test = load_or_create_test_data_for_model(df_segment, segment, mode, use_v2)
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

def analyze_all_models_with_optimization(use_v2=False):
    all_results = []
    
    # Determine the base path based on use_v2 flag
    data_suffix = "_v2" if use_v2 else ""
    base_models_path = f'/home/sagemaker-user/studio/src/new-rider-v3/models/decision_tree/all_features{data_suffix}'
    
    print(f"Analyzing models from: {base_models_path}")
    
    # Check if directory exists
    if not os.path.exists(base_models_path):
        print(f"ERROR: Directory {base_models_path} does not exist!")
        return pd.DataFrame(all_results)
    
    # Load data once at the beginning (following training script pattern)
    print("Loading data once for all analysis...")
    from utils.load_data import load_parquet_data
    df = load_parquet_data(use_v2)
    df = add_churned_indicator(df)
    
    # Apply the same preprocessing as training script
    if use_v2:
        assert 'rider_lyft_id' in df.columns, f"rider_lyft_id should be in columns when use_v2=True, but not found. Available columns: {[col for col in df.columns if 'session' in col]}"
        print(f"Verified: rider_lyft_id is in columns for V2 data")

    df = df.drop_duplicates(subset=['purchase_session_id'], keep='first')
    print(f"After deduplication: {len(df)} rows")
    
    # Assert that we have exactly 1 row per purchase_session_id
    assert df['purchase_session_id'].nunique() == len(df), f"Expected 1 row per purchase_session_id, but got {len(df)} rows for {df['purchase_session_id'].nunique()} unique purchase_session_ids"
    print(f"Verified: {len(df)} rows with {df['purchase_session_id'].nunique()} unique purchase_session_ids")

    CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                                'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id']
    df.drop(columns=CATEGORICAL_COLS_TO_DROP, inplace=True)
    
    # Note: rider_lyft_id is kept for rider counting functionality
    print(f"Verified: rider_lyft_id kept for analysis")
    
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
        required_cols = ['requested_ride_type', 'preselected_mode']
        df_segment = df_segment.dropna(subset=required_cols)
        print(f"Segment data shape: {df_segment.shape}")
        
        # Process each model for this segment
        for i, (file_path, mode, depth) in enumerate(models):
            print(f"\n{'='*60}")
            print(f"Analyzing: {segment_type}/{mode} (max_depth={depth})")
            print(f"{'='*60}")
            
            try:
                model, X_test, y_test, rider_ids_test = load_model_and_data(file_path, df_segment, use_v2)
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
                    'data_version': 'V2' if use_v2 else 'Original',
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

def create_optimized_analysis_plots(df, use_v2=False):
    """Create visualization plots for the optimized threshold analysis."""
    plt.style.use('default')
    sns.set_palette("YlGn")
    vmin, vmax = 0, 1
    
    # Add data version to title
    data_version = "V2" if use_v2 else "Original"
    
    fig, axes = plt.subplots(5, 3, figsize=(24, 30))
    fig.suptitle(f'Decision Tree Model Performance: Optimized vs Default Thresholds - {data_version} Data', fontsize=16, fontweight='bold')
    
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
    data_suffix = "_v2" if use_v2 else ""
    plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimized_analysis_plots_by_depth(df, use_v2=False):
    """Create separate visualization plots for each max_depth value."""
    plt.style.use('default')
    sns.set_palette("YlGn")
    vmin, vmax = 0, 1
    
    # Add data version to title
    data_version = "V2" if use_v2 else "Original"
    
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
        fig.suptitle(f'Decision Tree Model Performance: Optimized vs Default Thresholds - {data_version} Data (max_depth={max_depth})', fontsize=16, fontweight='bold')
        
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
        data_suffix = "_v2" if use_v2 else ""
        plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}_max_depth_{max_depth}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{data_suffix}_max_depth_{max_depth}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved for max_depth={max_depth}")

def print_optimized_analysis(df, use_v2=False):
    """Print detailed analysis of the optimized threshold results."""
    data_version = "V2" if use_v2 else "Original"
    
    print("=" * 80)
    print(f"DECISION TREE OPTIMIZED THRESHOLD ANALYSIS - {data_version} DATA")
    print("=" * 80)
    print(f"\nTotal models analyzed: {len(df)}")
    print(f"Data version: {data_version}")
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
    # Set to True to analyze V2 data, False for original data
    use_v2 = True
    
    Path('/home/sagemaker-user/studio/src/new-rider-v3/plots').mkdir(exist_ok=True, parents=True)
    print("Starting Decision Tree model analysis with threshold optimization...")
    df = analyze_all_models_with_optimization(use_v2)
    if len(df) > 0:
        print_optimized_analysis(df, use_v2)
        create_optimized_analysis_plots(df, use_v2)
        create_optimized_analysis_plots_by_depth(df, use_v2)
        
        # Save the analysis to CSV with data version suffix
        data_suffix = "_v2" if use_v2 else ""
        csv_path = f'/home/sagemaker-user/studio/src/new-rider-v3/reports/optimized_threshold_analysis_summary_decision_tree{data_suffix}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Analysis saved to '{csv_path}'")
        
        plot_suffix = "_v2" if use_v2 else ""
        print(f"📈 Plots saved to '/home/sagemaker-user/studio/src/new-rider-v3/plots/optimized_threshold_analysis_summary_decision_tree{plot_suffix}.pdf' and '.png'")
        print(f"📈 Separate plots created for each max_depth value: optimized_threshold_analysis_summary_decision_tree{plot_suffix}_max_depth_[3,5,10].pdf and .png")
    else:
        data_version = "V2" if use_v2 else "original"
        print(f"No Decision Tree models found to analyze for {data_version} data.") 