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

def create_plus_target(df):
    """Create binary target for Plus requests."""
    # Filter out sessions where Plus was preselected
    df = df[df['preselected_mode'] != 'plus'].copy()
    
    df['chose_plus'] = (df['requested_ride_type'] == 'plus').astype(int)
    return df

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
            results.append({
                'threshold': threshold,
                'threshold_pct': f"{threshold}" if equality else (f"{threshold:.2f}" if value_column == 'years_since_signup' else f"{threshold*100:.0f}%"),
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
    """Main analysis function."""
    print("Starting Plus vs Plus Lifetime Analysis")
    print(f"Segments: {segment_type_list}")
    print(f"Using {data_version.upper()} data")
    print("=" * 50)
    # Load data once
    print("Loading data...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    # Calculate totals for the whole dataset (for normalization)
    print("Calculating whole dataset totals for normalization...")
    df_whole = filter_by_segment(df, 'all')
    # Basic filtering for whole dataset
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df_whole = df_whole.dropna(subset=required_cols)
    df_whole = calculate_plus_percentage(df_whole)
    df_whole = create_plus_target(df_whole)
    total_sessions_all = len(df_whole)
    total_riders_all = df_whole['rider_lyft_id'].nunique()
    print(f"Whole dataset totals: {total_sessions_all:,} sessions, {total_riders_all:,} riders")
    # Process each segment type
    for segment_type in segment_type_list:
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_type}")
        print(f"{'='*60}")
        # Filter data for this segment
        df_segment = filter_by_segment(df, segment_type)
        # Basic filtering for segment
        df_segment = df_segment.dropna(subset=required_cols)
        df_segment = calculate_plus_percentage(df_segment)
        df_segment = create_plus_target(df_segment)
        # Analyze relationship
        # Remove analyze_plus_vs_plus_percentage and all calls to it
        # For percent_rides_plus_lifetime, call create_threshold_table directly and save the result

        # Create segment-specific output directories
        base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
        plots_dir = base_path / 'plots'
        reports_dir = base_path / f'reports/feature_threshold_analysis/segment_{segment_type}'
        reports_dir.mkdir(parents=True, exist_ok=True)

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

        print(f"\nAnalysis complete for segment: {segment_type}")
        print(f"Results saved to:")
        print(f"  - Plots: {plots_dir}")
        print(f"  - Reports: {reports_dir}")
        # Print key findings
        correlation = df_segment['percent_rides_plus_lifetime'].corr(df_segment['chose_plus'])
        print(f"Key Finding: Correlation between Plus choice and Plus percentage: {correlation:.3f}")

if __name__ == "__main__":
    # Configuration
    segment_type_list = ['airport', 'airport_dropoff', 'airport_pickup', 'all', 'churned']
    #segment_type_list = ['all']  # Uncomment to run only for all data
    # Set data version: 'original', 'v2', or 'v3'
    data_version = 'v3'
    main(segment_type_list, data_version) 