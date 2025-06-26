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
    """Prepare features and target for a specific mode (copied from training script)."""
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
    
    # Categorical columns to drop (high cardinality features that cause memory issues)
    CATEGORICAL_COLS_TO_DROP = ['purchase_session_id', 'candidate_product_key',
                                'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id']
    
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Use all categorical columns except the high cardinality ones
    feature_cols = [col for col in numeric_cols if col not in drop_cols] + [col for col in categorical_cols if (col not in drop_cols) and (col not in CATEGORICAL_COLS_TO_DROP)]
    X = df[feature_cols]
    y = df['target_diff_mode']
    
    print(f"Before get_dummies - X shape: {X.shape}")
    X = pd.get_dummies(X, drop_first=True)
    print(f"After get_dummies - X shape: {X.shape}")
    print("Data loaded and processed.")
    return X, y

def load_or_create_test_data_for_model(df, segment_type, mode):
    """Load test data from cache or create and save it for a specific model."""
    cache_key = f"test_data_{segment_type}_{mode}"
    cache_file = f'../data/{cache_key}.joblib'
    
    if os.path.exists(cache_file):
        print(f"Loading cached test data for {cache_key}...")
        try:
            cached_data = joblib.load(cache_file)
            X_test, y_test = cached_data
            print(f"✅ Loaded cached test data - Test: {X_test.shape}")
            return X_test, y_test
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Will recreate test data...")
    
    # Only load/process the full dataset if cache does not exist
    if df is None:
        print(f"Loading full dataset from S3 to create test data for {cache_key}...")
        from utils.load_data import load_parquet_data
        df = load_parquet_data()
        df = add_churned_indicator(df)
    
    print(f"Creating test data for {cache_key}...")
    
    # Filter data for this segment
    df_segment = filter_by_segment(df, segment_type)
    
    # Filter out rows with missing required columns
    required_cols = ['requested_ride_type', 'preselected_mode']
    df_segment = df_segment.dropna(subset=required_cols)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df_segment, mode)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test data to cache
    print(f"Saving test data to cache: {cache_file}")
    Path('../data').mkdir(exist_ok=True)
    joblib.dump((X_test, y_test), cache_file)
    
    print(f"✅ Created and cached test data - Test: {X_test.shape}")
    print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_test, y_test

def load_model_and_data(model_path, df=None):
    """Load a trained XGBoost model and prepare data for threshold optimization."""
    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Extract segment and mode from path
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
        
        # Load cached test data for this specific model
        X_test, y_test = load_or_create_test_data_for_model(df, segment, mode)
        
        return model, X_test, y_test
        
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None

def find_optimal_threshold(model, X_test, y_test):
    """Find optimal threshold for maximum F1 score (balance of precision and recall)."""
    print("Getting predictions on test data...")
    test_probs = model.predict_proba(X_test)[:, 1]
    
    print("Finding optimal threshold for maximum F1 score...")
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        test_preds = (test_probs >= threshold).astype(int)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Test F1 with optimal threshold: {best_f1:.4f}")
    print(f"Test precision with optimal threshold: {best_precision:.4f}")
    print(f"Test recall with optimal threshold: {best_recall:.4f}")
    
    # Get test predictions with both thresholds
    print("Getting predictions on test data...")
    
    # Default threshold (0.5)
    default_preds = (test_probs >= 0.5).astype(int)
    default_precision = precision_score(y_test, default_preds, zero_division=0)
    default_recall = recall_score(y_test, default_preds, zero_division=0)
    default_f1 = f1_score(y_test, default_preds, zero_division=0)
    default_accuracy = accuracy_score(y_test, default_preds)
    
    # Optimal threshold
    optimal_preds = (test_probs >= best_threshold).astype(int)
    optimal_precision = precision_score(y_test, optimal_preds, zero_division=0)
    optimal_recall = recall_score(y_test, optimal_preds, zero_division=0)
    optimal_f1 = f1_score(y_test, optimal_preds, zero_division=0)
    optimal_accuracy = accuracy_score(y_test, optimal_preds)
    
    # Calculate prediction distributions
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
        'test_support_negative': np.sum(y_test == 0)
    }
    
    return results

def analyze_all_models_with_optimization():
    """Analyze all XGBoost models with threshold optimization."""
    all_results = []
    
    # Only load the full dataset if needed (pass None to load_model_and_data)
    df = None
    
    # Walk through the models directory
    for root, _, files in os.walk('../models'):
        for file in files:
            if file != 'xgboost_model.joblib':
                continue
            
            file_path = os.path.join(root, file)
            
            # Extract segment and mode from path
            path_parts = file_path.split(os.sep)
            
            # Skip if not an XGBoost model with max_depth_10 and all_features
            if 'xg_boost' not in path_parts or 'max_depth_10' not in path_parts or 'all_features' not in path_parts:
                continue
            
            # Extract segment and mode
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
            
            print(f"\n{'='*60}")
            print(f"Analyzing: {segment}/{mode} (max_depth={depth})")
            print(f"{'='*60}")
            
            try:
                # Only load the full dataset if cache does not exist
                model, X_test, y_test = load_model_and_data(file_path, df)
                
                if model is None:
                    print(f"Skipping {file_path} due to loading error")
                    continue
                
                # Perform threshold optimization
                results = find_optimal_threshold(model, X_test, y_test)
                
                # Add metadata
                results.update({
                    'segment': segment,
                    'mode': mode,
                    'max_depth': int(depth),
                    'model_path': file_path
                })
                
                all_results.append(results)
                
                print(f"✅ Completed analysis for {segment}/{mode}")
                
            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {e}")
                continue
    
    return pd.DataFrame(all_results)

def create_optimized_analysis_plots(df):
    """Create visualization plots for the optimized threshold analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("YlGn")
    
    # Set a common color scale for all heatmaps except support
    vmin, vmax = 0, 1
    
    # Create a figure with multiple subplots (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(24, 26))
    fig.suptitle('XGBoost Model Performance: Optimized vs Default Thresholds', fontsize=16, fontweight='bold')

    # Row 1: Default Threshold Performance
    # 1. Default Precision heatmap
    ax1 = axes[0, 0]
    pivot_default_precision = df.pivot_table(index='mode', columns='segment', values='default_precision', aggfunc='mean')
    sns.heatmap(pivot_default_precision, annot=True, fmt='.3f', cmap='YlGn', ax=ax1, center=0.5, vmin=vmin, vmax=vmax)
    ax1.set_title('Default Threshold (0.5) - Precision')
    
    # 2. Default Recall heatmap
    ax2 = axes[0, 1]
    pivot_default_recall = df.pivot_table(index='mode', columns='segment', values='default_recall', aggfunc='mean')
    sns.heatmap(pivot_default_recall, annot=True, fmt='.3f', cmap='YlGn', ax=ax2, center=0.5, vmin=vmin, vmax=vmax)
    ax2.set_title('Default Threshold (0.5) - Recall')
    
    # 3. Default F1 Score heatmap
    ax3 = axes[0, 2]
    pivot_default_f1 = df.pivot_table(index='mode', columns='segment', values='default_f1', aggfunc='mean')
    sns.heatmap(pivot_default_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax3, center=0.5, vmin=vmin, vmax=vmax)
    ax3.set_title('Default Threshold (0.5) - F1 Score')
    
    # Row 2: Optimized Threshold Performance
    # 4. Optimized Precision heatmap
    ax4 = axes[1, 0]
    pivot_optimal_precision = df.pivot_table(index='mode', columns='segment', values='optimal_precision', aggfunc='mean')
    sns.heatmap(pivot_optimal_precision, annot=True, fmt='.3f', cmap='YlGn', ax=ax4, center=0.5, vmin=vmin, vmax=vmax)
    ax4.set_title('Optimized Threshold - Precision')
    
    # 5. Optimized Recall heatmap
    ax5 = axes[1, 1]
    pivot_optimal_recall = df.pivot_table(index='mode', columns='segment', values='optimal_recall', aggfunc='mean')
    sns.heatmap(pivot_optimal_recall, annot=True, fmt='.3f', cmap='YlGn', ax=ax5, center=0.5, vmin=vmin, vmax=vmax)
    ax5.set_title('Optimized Threshold - Recall')
    
    # 6. Optimized F1 Score heatmap
    ax6 = axes[1, 2]
    pivot_optimal_f1 = df.pivot_table(index='mode', columns='segment', values='optimal_f1', aggfunc='mean')
    sns.heatmap(pivot_optimal_f1, annot=True, fmt='.3f', cmap='YlGn', ax=ax6, center=0.5, vmin=vmin, vmax=vmax)
    ax6.set_title('Optimized Threshold - F1 Score')
    
    # Row 3: Improvements and Analysis
    # 7. Precision Improvement heatmap
    ax7 = axes[2, 0]
    df['precision_improvement'] = df['optimal_precision'] - df['default_precision']
    pivot_precision_improvement = df.pivot_table(index='mode', columns='segment', values='precision_improvement', aggfunc='mean')
    sns.heatmap(pivot_precision_improvement, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7, center=0)
    ax7.set_title('Precision Improvement (Optimal - Default)')
    
    # 8. F1 Score Improvement heatmap
    ax8 = axes[2, 1]
    df['f1_improvement'] = df['optimal_f1'] - df['default_f1']
    pivot_f1_improvement = df.pivot_table(index='mode', columns='segment', values='f1_improvement', aggfunc='mean')
    sns.heatmap(pivot_f1_improvement, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax8, center=0)
    ax8.set_title('F1 Score Improvement (Optimal - Default)')
    
    # 9. Optimal Threshold Values heatmap
    ax9 = axes[2, 2]
    pivot_optimal_threshold = df.pivot_table(index='mode', columns='segment', values='optimal_threshold', aggfunc='mean')
    sns.heatmap(pivot_optimal_threshold, annot=True, fmt='.3f', cmap='YlGn', ax=ax9, vmin=0.3, vmax=0.9)
    ax9.set_title('Optimal Threshold Values')

    # Row 4: Positive Prediction Ratios (relative to total samples)
    total_samples = 122925  # Always use the global total
    # 10. Default Positive Prediction Ratio heatmap (global denominator)
    ax10 = axes[3, 0]
    pivot_default_pos_count = df.pivot_table(index='mode', columns='segment', values='default_positive_preds', aggfunc='mean')
    pivot_default_pos_ratio_global = (pivot_default_pos_count / total_samples) * 100
    annot_default = pivot_default_pos_ratio_global.round(2).astype(str) + '% (' + pivot_default_pos_count.round(0).astype(int).astype(str) + ')'
    sns.heatmap(pivot_default_pos_ratio_global, annot=annot_default, fmt='', cmap='YlGn', ax=ax10, vmin=0, vmax=100)
    ax10.set_title('Default Threshold - % Predicted Positive (Global)')
    
    # 11. Optimized Positive Prediction Ratio heatmap (global denominator)
    ax11 = axes[3, 1]
    pivot_optimal_pos_count = df.pivot_table(index='mode', columns='segment', values='optimal_positive_preds', aggfunc='mean')
    pivot_optimal_pos_ratio_global = (pivot_optimal_pos_count / total_samples) * 100
    annot_optimal = pivot_optimal_pos_ratio_global.round(2).astype(str) + '% (' + pivot_optimal_pos_count.round(0).astype(int).astype(str) + ')'
    sns.heatmap(pivot_optimal_pos_ratio_global, annot=annot_optimal, fmt='', cmap='YlGn', ax=ax11, vmin=0, vmax=100)
    ax11.set_title('Optimized Threshold - % Predicted Positive (Global)')
    
    # 12. Empty plot (for layout)
    ax12 = axes[3, 2]
    ax12.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../plots/optimized_threshold_analysis_summary_xgboost.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../plots/optimized_threshold_analysis_summary_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_optimized_analysis(df):
    """Print detailed analysis of the optimized threshold results."""
    print("=" * 80)
    print("XGBOOST OPTIMIZED THRESHOLD ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal models analyzed: {len(df)}")
    print(f"Segments: {df['segment'].unique()}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Max depths: {df['max_depth'].unique()}")
    
    # Sanity check for total samples
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
    
    # Sort by optimized F1 score
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
    
    # Find best overall performer
    best_overall = df.loc[df['optimal_f1'].idxmax()]
    
    print(f"\n🏆 BEST OVERALL PERFORMER:")
    print(f"  {best_overall['segment']}/{best_overall['mode']}")
    print(f"  - Optimal F1: {best_overall['optimal_f1']:.3f}")
    print(f"  - Optimal Threshold: {best_overall['optimal_threshold']:.3f}")
    print(f"  - Precision: {best_overall['optimal_precision']:.3f}")
    print(f"  - Recall: {best_overall['optimal_recall']:.3f}")
    print(f"  - Prediction Ratio: {best_overall['optimal_positive_ratio']:.1f}%")
    
    # Models with high precision after optimization
    high_precision = df[df['optimal_precision'] > 0.7]
    if len(high_precision) > 0:
        print(f"\n🎯 MODELS WITH HIGH PRECISION (>0.7) AFTER OPTIMIZATION:")
        for _, row in high_precision.iterrows():
            print(f"  - {row['segment']}/{row['mode']}: {row['optimal_precision']:.3f} precision ({row['optimal_positive_ratio']:.1f}% predicted positive)")
    
    # Models with significant F1 improvement
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
    Path('../plots').mkdir(exist_ok=True)
    print("Starting XGBoost model analysis with threshold optimization...")
    df = analyze_all_models_with_optimization()
    if len(df) > 0:
        print_optimized_analysis(df)
        create_optimized_analysis_plots(df)
        df.to_csv('../reports/optimized_threshold_analysis_summary_xgboost.csv', index=False)
        print(f"\n📊 Analysis saved to '../reports/optimized_threshold_analysis_summary_xgboost.csv'")
        print(f"📈 Plots saved to '../plots/optimized_threshold_analysis_summary_xgboost.pdf' and '../plots/optimized_threshold_analysis_summary_xgboost.png'")
    else:
        print("No XGBoost models found to analyze.") 