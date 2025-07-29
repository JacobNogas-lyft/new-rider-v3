import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_parquet_data
from pathlib import Path
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

def filter_curbside_pickup_data(df):
    """
    Filter dataframe to only include curbside pickup sessions at LAX, ORD, and SFO airports.
    
    Args:
        df (pandas.DataFrame): Input dataframe
    
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    if 'pickup_airport_code' not in df.columns:
        print("Warning: 'pickup_airport_code' column not found in data. No filtering applied.")
        return df.copy()
    
    curbside_mask = df['pickup_airport_code'].isin(['LAX', 'ORD', 'SFO'])
    filtered_df = df[curbside_mask].copy()
    print(f"Curbside pickup sessions (LAX/ORD/SFO): {len(filtered_df)} rows (from {len(df)} total)")
    return filtered_df

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

def load_and_prepare_data(segment_type='all', data_version='v2'):
    """Load and prepare data for Plus vs standard_saver choice analysis."""
    print(f"Loading and preparing data for segment: {segment_type} (data_version={data_version})...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    # Filter to curbside pickup data first
    df = filter_curbside_pickup_data(df)
    # Filter by segment
    df = filter_by_segment(df, segment_type)
    # Basic filtering
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df = df.dropna(subset=required_cols)
    print(f"Data loaded: {len(df)} rows")
    return df

def calculate_plus_percentage(df):
    """Calculate percentage of lifetime rides that were Plus."""
    # Handle division by zero by replacing 0 with NaN, then filling with 0
    df['percent_rides_plus_lifetime'] = (
        df['rides_plus_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    # Clip to 0-1 range (in case of data issues)
    #df['percent_rides_plus_lifetime'] = df['percent_rides_plus_lifetime'].clip(0, 1)
    
    return df

def calculate_premium_percentage(df):
    """Calculate percentage of lifetime rides that were Premium."""
    # Check if rides_premium_lifetime column exists
    if 'rides_premium_lifetime' in df.columns:
        # Handle division by zero by replacing 0 with NaN, then filling with 0
        df['Percent_rides_premium_lifetime'] = (
            df['rides_premium_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
        ).fillna(0)
        
        # Clip to 0-1 range (in case of data issues)
        df['Percent_rides_premium_lifetime'] = df['Percent_rides_premium_lifetime'].clip(0, 1)
    else:
        print("Warning: 'rides_premium_lifetime' column not found. Setting Percent_rides_premium_lifetime to 0.")
        df['Percent_rides_premium_lifetime'] = 0
    
    return df

def create_plus_target(df):
    """Create binary target for Plus requests."""
    # Filter out sessions where Plus was preselected
    df = df[df['preselected_mode'] != 'plus'].copy()
    
    df['chose_plus'] = (df['requested_ride_type'] == 'plus').astype(int)
    return df

def calculate_feature_statistics(df, feature_name):
    """
    Calculate descriptive statistics for a numerical feature.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_name (str): Name of the feature to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if feature_name not in df.columns:
        print(f"Warning: '{feature_name}' column not found in data.")
        return None
    
    data = df[feature_name].dropna()
    if len(data) == 0:
        print(f"Warning: No valid data found for '{feature_name}'.")
        return None
    
    stats = {
        'count': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
        'q90': data.quantile(0.90),
        'q95': data.quantile(0.95),
        'q99': data.quantile(0.99),
        'non_null_pct': (len(data) / len(df)) * 100
    }
    
    return stats

def print_feature_statistics(stats, feature_name):
    """
    Print formatted statistics for a feature.
    
    Args:
        stats (dict): Statistics dictionary
        feature_name (str): Name of the feature
    """
    if stats is None:
        return
    
    print(f"\n=== Descriptive Statistics for {feature_name} ===")
    print(f"Count (non-null): {stats['count']:,}")
    print(f"Non-null percentage: {stats['non_null_pct']:.1f}%")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Min: {stats['min']:.2f}")
    print(f"Max: {stats['max']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"25th percentile: {stats['q25']:.2f}")
    print(f"75th percentile: {stats['q75']:.2f}")
    print(f"90th percentile: {stats['q90']:.2f}")
    print(f"95th percentile: {stats['q95']:.2f}")
    print(f"99th percentile: {stats['q99']:.2f}")

def calculate_special_rider_statistics(df_full, df_curbside):
    """
    Calculate special statistics about rider percentages and purchase session percentages.
    
    Args:
        df_full (pandas.DataFrame): Full dataset before curbside filtering
        df_curbside (pandas.DataFrame): Curbside-filtered dataset
    
    Returns:
        dict: Dictionary containing special statistics
    """
    total_riders_full = df_full['rider_lyft_id'].nunique()
    total_sessions_full = len(df_full)
    
    special_stats = {}
    
    # 1. Calculate % of all riders who have LAX/ORD/SFO sessions
    curbside_riders = df_curbside['rider_lyft_id'].nunique()
    special_stats['pct_riders_lax_ord_sfo'] = (curbside_riders / total_riders_full) * 100
    
    # 1b. Calculate % of all sessions that are LAX/ORD/SFO sessions
    curbside_sessions = len(df_curbside)
    special_stats['pct_sessions_lax_ord_sfo'] = (curbside_sessions / total_sessions_full) * 100
    
    # 2. Calculate % of LAX/ORD/SFO riders who have Black ATF
    # Check if Black is ATF by looking for 'lux' in rank_1, rank_2, rank_3, or rank_4
    rank_columns = ['rank_1', 'rank_2', 'rank_3', 'rank_4']
    available_rank_columns = [col for col in rank_columns if col in df_curbside.columns]
    
    if available_rank_columns and curbside_riders > 0:
        # Create mask for sessions where any rank contains 'lux'
        lux_mask = False
        for col in available_rank_columns:
            lux_mask |= (df_curbside[col] == 'lux')
        
        black_atf_riders = df_curbside[lux_mask]['rider_lyft_id'].nunique()
        special_stats['pct_lax_ord_sfo_riders_with_black_atf'] = (black_atf_riders / curbside_riders) * 100
    else:
        special_stats['pct_lax_ord_sfo_riders_with_black_atf'] = 0.0
        if not available_rank_columns:
            print("Warning: No rank columns found for Black ATF calculation")
    
    # 3. Calculate % of all riders who have LAX/ORD/SFO session AND Lux Final Price Diff <= 44$
    if 'lux_final_price_diff_wrt_standard_major_currency' in df_curbside.columns:
        combined_riders = df_curbside[
            df_curbside['lux_final_price_diff_wrt_standard_major_currency'].notna() & 
            (df_curbside['lux_final_price_diff_wrt_standard_major_currency'] <= 44.0)
        ]['rider_lyft_id'].nunique()
        special_stats['pct_riders_lax_ord_sfo_and_lux_price_lte_44'] = (combined_riders / total_riders_full) * 100
    else:
        # If lux price column not available, use LAX/ORD/SFO sessions as fallback
        special_stats['pct_riders_lax_ord_sfo_and_lux_price_lte_44'] = special_stats['pct_riders_lax_ord_sfo']
    
    # 4. Calculate % of purchase sessions Black ATF
    if available_rank_columns:
        # Create mask for sessions where any rank contains 'lux'
        lux_mask = False
        for col in available_rank_columns:
            lux_mask |= (df_full[col] == 'lux')
        
        black_atf_sessions = len(df_full[lux_mask])
        special_stats['pct_sessions_black_atf'] = (black_atf_sessions / total_sessions_full) * 100
    else:
        special_stats['pct_sessions_black_atf'] = 0.0
    
    # 5. Calculate % of purchase sessions where LAX/ORD/SFO + Lux price gap <= 44$
    if 'lux_final_price_diff_wrt_standard_major_currency' in df_curbside.columns:
        combined_sessions = len(df_curbside[
            df_curbside['lux_final_price_diff_wrt_standard_major_currency'].notna() & 
            (df_curbside['lux_final_price_diff_wrt_standard_major_currency'] <= 44.0)
        ])
        special_stats['pct_sessions_lax_ord_sfo_and_lux_price_lte_44'] = (combined_sessions / total_sessions_full) * 100
    else:
        # If lux price column not available, use LAX/ORD/SFO sessions as fallback
        curbside_sessions = len(df_curbside)
        special_stats['pct_sessions_lax_ord_sfo_and_lux_price_lte_44'] = (curbside_sessions / total_sessions_full) * 100
    
    return special_stats

def save_feature_statistics(stats, feature_name, output_path, special_stats=None):
    """
    Save feature statistics to a CSV file.
    
    Args:
        stats (dict): Statistics dictionary
        feature_name (str): Name of the feature
        output_path (Path): Path to save the CSV file
        special_stats (dict): Special rider statistics
    """
    if stats is None:
        return
    
    # Create a dataframe with the statistics
    stats_row = {
        'feature_name': feature_name,
        'count': stats['count'],
        'non_null_pct': stats['non_null_pct'],
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max'],
        'median': stats['median'],
        'q25': stats['q25'],
        'q75': stats['q75'],
        'q90': stats['q90'],
        'q95': stats['q95'],
        'q99': stats['q99']
    }
    
    # Add special statistics if provided
    if special_stats:
        for key, value in special_stats.items():
            stats_row[key] = value
    
    stats_df = pd.DataFrame([stats_row])
    
    stats_df.to_csv(output_path, index=False)
    print(f"Saved statistics for {feature_name} to: {output_path}")
    
    # Print special stats if available
    if special_stats:
        print("Special rider statistics:")
        for key, value in special_stats.items():
            print(f"  {key}: {value:.2f}%")

def create_threshold_table(df, total_sessions_all, total_riders_all, thresholds=None, value_column='percent_rides_plus_lifetime', equality=False):
    """Create a table showing probability of choosing Plus, standard_saver, premium, and lux for sessions where value_column > threshold (or == threshold if equality=True)."""
    # Default thresholds for percent_rides_plus_lifetime
    if thresholds is None:
        thresholds = [-0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    results = []
    for threshold in thresholds:
        if equality:
            mask = df[value_column] == threshold
        else:
            mask = df[value_column] > threshold
        subset = df[mask]
        if len(subset) > 0:
            plus_rate = subset['chose_plus'].mean()
            plus_sessions = subset['chose_plus'].sum()
            standard_saver_rate = (subset['requested_ride_type'] == 'standard_saver').mean()
            standard_saver_sessions = (subset['requested_ride_type'] == 'standard_saver').sum()
            premium_rate = (subset['requested_ride_type'] == 'premium').mean()
            lux_rate = (subset['requested_ride_type'] == 'lux').mean()
            fastpass_rate = (subset['requested_ride_type'] == 'fastpass').mean()
            total_sessions = len(subset)
            distinct_riders = subset['rider_lyft_id'].nunique()
            sessions_pct = (total_sessions / total_sessions_all) * 100
            riders_pct = (distinct_riders / total_riders_all) * 100
            # Format threshold display based on column type
            if equality:
                threshold_display = f"{threshold}"
            elif value_column == 'years_since_signup':
                threshold_display = f"{threshold:.2f}"
            elif value_column == 'haversine_dist_km':
                threshold_display = f"{threshold:.1f} km"
            elif value_column == 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes':
                threshold_display = f"{threshold:.1f} min"
            elif value_column == 'lux_final_price_diff_wrt_standard_major_currency':
                threshold_display = f"{threshold:.2f}"
            else:
                threshold_display = f"{threshold*100:.0f}%"
            
            results.append({
                'threshold': threshold,
                'threshold_pct': threshold_display,
                'total_sessions': total_sessions,
                'sessions_pct': f"{sessions_pct:.1f}%",
                'distinct_riders': distinct_riders,
                'riders_pct': f"{riders_pct:.1f}%",
                'plus_sessions': plus_sessions,
                'plus_probability': plus_rate,
                'plus_probability_pct': f"{plus_rate*100:.1f}%",
                'standard_saver_sessions': standard_saver_sessions,
                'standard_saver_rate': standard_saver_rate,
                'standard_saver_probability_pct': f"{standard_saver_rate*100:.1f}%",
                'premium_rate': premium_rate,
                'lux_rate': lux_rate,
                'fastpass_rate': fastpass_rate
            })
    threshold_df = pd.DataFrame(results)
    return threshold_df

def main(segment_type_list=['all'], data_version='v2'):
    """Main analysis function for curbside pickup analysis."""
    print("Starting Plus vs Plus Lifetime Analysis - CURBSIDE PICKUP (LAX/ORD/SFO)")
    print(f"Segments: {segment_type_list}")
    print(f"Using {data_version.upper()} data")
    print("=" * 60)
    # Load data once
    print("Loading data...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    
    # Calculate totals for the FULL dataset BEFORE curbside filtering (for normalization)
    print("Calculating FULL dataset totals for normalization (before curbside filtering)...")
    df_full = filter_by_segment(df, 'all')  # No curbside filtering yet
    # Basic filtering for full dataset
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df_full = df_full.dropna(subset=required_cols)
    df_full = calculate_plus_percentage(df_full)
    df_full = calculate_premium_percentage(df_full)
    df_full = create_plus_target(df_full)
    total_sessions_all = len(df_full)
    total_riders_all = df_full['rider_lyft_id'].nunique()
    print(f"Full dataset totals (BEFORE curbside filtering): {total_sessions_all:,} sessions, {total_riders_all:,} riders")
    
    # NOW filter to curbside pickup data for analysis (LAX/ORD/SFO)
    df = filter_curbside_pickup_data(df)
    # Process each segment type
    for segment_type in segment_type_list:
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_type} (CURBSIDE PICKUP)")
        print(f"{'='*60}")
        # Filter data for this segment
        df_segment = filter_by_segment(df, segment_type)
        # Basic filtering for segment
        df_segment = df_segment.dropna(subset=required_cols)
        df_segment = calculate_plus_percentage(df_segment)
        df_segment = calculate_premium_percentage(df_segment)
        df_segment = create_plus_target(df_segment)
        
        # Create segment-specific output directories
        base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
        plots_dir = base_path / 'plots'
        reports_dir = base_path / f'reports/curbside_pickup_threshold_analysis/segment_{segment_type}'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate special rider statistics (once per segment)
        print(f"\n=== Special Rider Statistics ===")
        special_stats_general = calculate_special_rider_statistics(df_full, df_segment)
        print(f"% of all riders with LAX/ORD/SFO sessions: {special_stats_general['pct_riders_lax_ord_sfo']:.2f}%")
        print(f"% of all sessions that are LAX/ORD/SFO sessions: {special_stats_general['pct_sessions_lax_ord_sfo']:.2f}%")
        print(f"% of LAX/ORD/SFO riders with Black ATF: {special_stats_general['pct_lax_ord_sfo_riders_with_black_atf']:.2f}%")
        print(f"% of all riders with LAX/ORD/SFO sessions AND Lux Final Price Diff <= 44$: {special_stats_general['pct_riders_lax_ord_sfo_and_lux_price_lte_44']:.2f}%")
        print(f"% of purchase sessions Black ATF: {special_stats_general['pct_sessions_black_atf']:.2f}%")
        print(f"% of purchase sessions where LAX/ORD/SFO + Lux price gap <= 44$: {special_stats_general['pct_sessions_lax_ord_sfo_and_lux_price_lte_44']:.2f}%")
        
        # Save all special statistics to a file
        special_stats_rows = [
            {
                'segment': segment_type,
                'metric': 'pct_riders_lax_ord_sfo',
                'value': special_stats_general['pct_riders_lax_ord_sfo'],
                'description': 'Percentage of all riders who have LAX/ORD/SFO sessions'
            },
            {
                'segment': segment_type,
                'metric': 'pct_sessions_lax_ord_sfo',
                'value': special_stats_general['pct_sessions_lax_ord_sfo'],
                'description': 'Percentage of all sessions that are LAX/ORD/SFO sessions'
            },
            {
                'segment': segment_type,
                'metric': 'pct_lax_ord_sfo_riders_with_black_atf',
                'value': special_stats_general['pct_lax_ord_sfo_riders_with_black_atf'],
                'description': 'Percentage of LAX/ORD/SFO riders who have Black ATF'
            },
            {
                'segment': segment_type,
                'metric': 'pct_riders_lax_ord_sfo_and_lux_price_lte_44',
                'value': special_stats_general['pct_riders_lax_ord_sfo_and_lux_price_lte_44'],
                'description': 'Percentage of all riders who have LAX/ORD/SFO session AND Lux Final Price Diff <= 44$'
            },
            {
                'segment': segment_type,
                'metric': 'pct_sessions_black_atf',
                'value': special_stats_general['pct_sessions_black_atf'],
                'description': 'Percentage of purchase sessions Black ATF'
            },
            {
                'segment': segment_type,
                'metric': 'pct_sessions_lax_ord_sfo_and_lux_price_lte_44',
                'value': special_stats_general['pct_sessions_lax_ord_sfo_and_lux_price_lte_44'],
                'description': 'Percentage of purchase sessions where LAX/ORD/SFO + Lux price gap <= 44$'
            }
        ]
        special_stats_df = pd.DataFrame(special_stats_rows)
        special_stats_df.to_csv(reports_dir / 'special_rider_statistics.csv', index=False)
        print(f"Saved special rider statistics to: {reports_dir / 'special_rider_statistics.csv'}")
        # Analyze relationship
        # Remove analyze_plus_vs_plus_percentage and all calls to it
        # For percent_rides_plus_lifetime, call create_threshold_table directly and save the result

        # --- Threshold-style table for percent_rides_plus_lifetime ---
        print(f"\n=== Threshold-style table for percent_rides_plus_lifetime ===")
        threshold_df = create_threshold_table(
            df_segment,
            value_column='percent_rides_plus_lifetime',
            total_sessions_all=total_sessions_all,
            total_riders_all=total_riders_all
        )
        threshold_df.to_csv(reports_dir / f'threshold_by_percent_rides_plus_lifetime.csv', index=False)
        print(f"Saved threshold-style table for percent_rides_plus_lifetime to: {reports_dir / f'threshold_by_percent_rides_plus_lifetime.csv'}")

        # --- Threshold-style tables for airline_destination, airline_pickup, days_since_signup (0.5 year bins) ---
        for feature in ['airline_destination', 'airline_pickup']:
            if feature in df_segment.columns:
                print(f"\n=== Threshold-style table for {feature} ===")
                thresholds = df_segment[feature].dropna().unique()
                threshold_df = create_threshold_table(
                    df_segment,
                    thresholds=thresholds,
                    value_column=feature,
                    equality=True,
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                threshold_df.to_csv(reports_dir / f'threshold_by_{feature}.csv', index=False)
                print(f"Saved threshold-style table for {feature} to: {reports_dir / f'threshold_by_{feature}.csv'}")
            else:
                print(f"Column '{feature}' not found in data for this segment. Skipping threshold table for {feature}.")

        # Days since signup threshold table: years_since_signup > t (0.5 year increments)
        if 'days_since_signup' in df_segment.columns:
            print("\n=== Threshold-style table for years_since_signup > t (0.5 year increments) ===")
            # Convert days to years
            df_segment['years_since_signup'] = df_segment['days_since_signup'] / 365.25
            max_years = df_segment['years_since_signup'].max()
            thresholds = np.arange(0, max_years + 0.5, 0.5)
            threshold_df = create_threshold_table(
                df_segment,
                thresholds=thresholds,
                value_column='years_since_signup',
                total_sessions_all=total_sessions_all,
                total_riders_all=total_riders_all
            )
            threshold_df.to_csv(reports_dir / f'threshold_by_years_since_signup.csv', index=False)
            print(f"Saved threshold-style table for years_since_signup thresholds to: {reports_dir / f'threshold_by_years_since_signup.csv'}")
        else:
            print("Column 'days_since_signup' not found in data for this segment. Skipping threshold table for days_since_signup.")

        # Haversine distance threshold table: haversine_dist_meters_bw_dest_and_top_geohash6 > t
        if 'haversine_dist_meters_bw_dest_and_top_geohash6' in df_segment.columns:
            print("\n=== Threshold-style table for haversine_dist_meters_bw_dest_and_top_geohash6 > t ===")
            # Convert meters to kilometers for better readability
            df_segment['haversine_dist_km'] = df_segment['haversine_dist_meters_bw_dest_and_top_geohash6'] / 1000
            
            # Create thresholds based on data distribution
            dist_data = df_segment['haversine_dist_km'].dropna()
            if len(dist_data) > 0:
                max_dist = dist_data.max()
                min_dist = dist_data.min()
                
                # Create thresholds: 0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50km, then percentiles for higher values
                base_thresholds = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
                
                # Add percentile-based thresholds for higher distances
                if max_dist > 50:
                    percentiles = [75, 80, 85, 90, 95, 99]
                    percentile_thresholds = [np.percentile(dist_data, p) for p in percentiles]
                    thresholds = base_thresholds + percentile_thresholds
                else:
                    thresholds = base_thresholds
                
                # Filter thresholds to be within data range and remove duplicates
                thresholds = sorted(set([t for t in thresholds if t <= max_dist]))
                
                threshold_df = create_threshold_table(
                    df_segment,
                    thresholds=thresholds,
                    value_column='haversine_dist_km',
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                threshold_df.to_csv(reports_dir / f'threshold_by_haversine_dist_km.csv', index=False)
                print(f"Saved threshold-style table for haversine distance thresholds to: {reports_dir / f'threshold_by_haversine_dist_km.csv'}")
                print(f"Distance range: {min_dist:.2f} - {max_dist:.2f} km")
            else:
                print("No valid distance data found in this segment.")
        else:
            print("Column 'haversine_dist_meters_bw_dest_and_top_geohash6' not found in data for this segment. Skipping distance threshold table.")

        # --- Plus rate by signup_year, airline_destination, airline_pickup using create_threshold_table ---
        for feature in ['signup_year', 'airline_destination', 'airline_pickup']:
            if feature in df_segment.columns:
                print(f"\n=== Plus rate by {feature} ===")
                if feature == 'signup_year':
                    thresholds = np.sort(df_segment[feature].dropna().astype(int).unique())
                else:
                    thresholds = df_segment[feature].dropna().unique()
                threshold_df = create_threshold_table(
                    df_segment,
                    thresholds=thresholds,
                    value_column=feature,
                    equality=True,
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                threshold_df.to_csv(reports_dir / f'threshold_by_{feature}.csv', index=False)
                print(f"Saved plus rate by {feature} to: {reports_dir / f'threshold_by_{feature}.csv'}")
            else:
                print(f"Column '{feature}' not found in data for this segment. Skipping {feature} analysis.")

        # --- Threshold-style table for Percent_rides_premium_lifetime ---
        print(f"\n=== Threshold-style table for Percent_rides_premium_lifetime ===")
        threshold_df = create_threshold_table(
            df_segment,
            value_column='Percent_rides_premium_lifetime',
            total_sessions_all=total_sessions_all,
            total_riders_all=total_riders_all
        )
        threshold_df.to_csv(reports_dir / f'threshold_by_Percent_rides_premium_lifetime.csv', index=False)
        print(f"Saved threshold-style table for Percent_rides_premium_lifetime to: {reports_dir / f'threshold_by_Percent_rides_premium_lifetime.csv'}")

        # --- Threshold-style table for lux_pin_eta_diff_wrt_standard_pin_eta_minutes ---
        if 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes' in df_segment.columns:
            # Calculate and display descriptive statistics first
            eta_stats = calculate_feature_statistics(df_segment, 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes')
            print_feature_statistics(eta_stats, 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes')
            save_feature_statistics(eta_stats, 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes', 
                                  reports_dir / 'descriptive_stats_lux_pin_eta_diff_minutes.csv',
                                  special_stats=special_stats_general)
            
            print(f"\n=== Threshold-style table for lux_pin_eta_diff_wrt_standard_pin_eta_minutes ===")
            eta_data = df_segment['lux_pin_eta_diff_wrt_standard_pin_eta_minutes'].dropna()
            if len(eta_data) > 0:
                max_eta = eta_data.max()
                min_eta = eta_data.min()
                
                # Create thresholds based on time differences: 0, 1, 2, 3, 5, 10, 15, 20, 30 minutes, then percentiles
                base_thresholds = [0, 1, 2, 3, 5, 10, 15, 20, 30]
                
                # Add percentile-based thresholds for higher values
                if max_eta > 30:
                    percentiles = [75, 80, 85, 90, 95, 99]
                    percentile_thresholds = [np.percentile(eta_data, p) for p in percentiles]
                    thresholds = base_thresholds + percentile_thresholds
                else:
                    thresholds = base_thresholds
                
                # Filter thresholds to be within data range and remove duplicates
                thresholds = sorted(set([t for t in thresholds if t <= max_eta and t >= min_eta]))
                
                threshold_df = create_threshold_table(
                    df_segment,
                    thresholds=thresholds,
                    value_column='lux_pin_eta_diff_wrt_standard_pin_eta_minutes',
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                threshold_df.to_csv(reports_dir / f'threshold_by_lux_pin_eta_diff_minutes.csv', index=False)
                print(f"Saved threshold-style table for lux ETA diff to: {reports_dir / f'threshold_by_lux_pin_eta_diff_minutes.csv'}")
                print(f"ETA diff range: {min_eta:.2f} - {max_eta:.2f} minutes")
            else:
                print("No valid ETA diff data found in this segment.")
        else:
            print("Column 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes' not found in data for this segment. Skipping ETA diff threshold table.")

        # --- Threshold-style table for lux_final_price_diff_wrt_standard_major_currency ---
        if 'lux_final_price_diff_wrt_standard_major_currency' in df_segment.columns:
            # Calculate and display descriptive statistics first
            price_stats = calculate_feature_statistics(df_segment, 'lux_final_price_diff_wrt_standard_major_currency')
            print_feature_statistics(price_stats, 'lux_final_price_diff_wrt_standard_major_currency')
            
            save_feature_statistics(price_stats, 'lux_final_price_diff_wrt_standard_major_currency', 
                                  reports_dir / 'descriptive_stats_lux_final_price_diff_currency.csv',
                                  special_stats=special_stats_general)
            
            print(f"\n=== Threshold-style table for lux_final_price_diff_wrt_standard_major_currency ===")
            price_data = df_segment['lux_final_price_diff_wrt_standard_major_currency'].dropna()
            if len(price_data) > 0:
                max_price = price_data.max()
                min_price = price_data.min()
                
                # Create thresholds based on price differences: 0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, then percentiles
                base_thresholds = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
                
                # Add percentile-based thresholds for higher values
                if max_price > 50:
                    percentiles = [75, 80, 85, 90, 95, 99]
                    percentile_thresholds = [np.percentile(price_data, p) for p in percentiles]
                    thresholds = base_thresholds + percentile_thresholds
                else:
                    thresholds = base_thresholds
                
                # Filter thresholds to be within data range and remove duplicates
                thresholds = sorted(set([t for t in thresholds if t <= max_price and t >= min_price]))
                
                threshold_df = create_threshold_table(
                    df_segment,
                    thresholds=thresholds,
                    value_column='lux_final_price_diff_wrt_standard_major_currency',
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                threshold_df.to_csv(reports_dir / f'threshold_by_lux_final_price_diff_currency.csv', index=False)
                print(f"Saved threshold-style table for lux price diff to: {reports_dir / f'threshold_by_lux_final_price_diff_currency.csv'}")
                print(f"Price diff range: {min_price:.2f} - {max_price:.2f} currency units")
            else:
                print("No valid price diff data found in this segment.")
        else:
            print("Column 'lux_final_price_diff_wrt_standard_major_currency' not found in data for this segment. Skipping price diff threshold table.")

        print(f"\nAnalysis complete for segment: {segment_type}")
        print(f"Results saved to:")
        print(f"  - Plots: {plots_dir}")
        print(f"  - Reports: {reports_dir}")
        # Print key findings
        correlation = df_segment['percent_rides_plus_lifetime'].corr(df_segment['chose_plus'])
        print(f"Key Finding: Correlation between Plus choice and Plus percentage: {correlation:.3f}")

if __name__ == "__main__":
    # Configuration
    segment_type_list = ['airport_pickup']
    #segment_type_list = ['all']  # Uncomment to run only for all data
    # Set data version: 'original', 'v2', or 'v3'
    data_version = 'v3'
    main(segment_type_list, data_version) 