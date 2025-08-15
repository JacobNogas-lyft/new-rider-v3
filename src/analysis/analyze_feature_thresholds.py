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

def calculate_premium_percentage(df):
    """Calculate percentage of lifetime rides that were Premium."""
    # Check if rides_premium_lifetime column exists
    if 'rides_premium_lifetime' in df.columns:
        # Handle division by zero by replacing 0 with NaN, then filling with 0
        df['Percent_rides_premium_lifetime'] = (
            df['rides_premium_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
        ).fillna(0)
        
        # Assert that Percent_rides_premium_lifetime is between 0 and 1
        assert ((df['Percent_rides_premium_lifetime'] >= 0) & (df['Percent_rides_premium_lifetime'] <= 1)).all(), "Percent_rides_premium_lifetime values must be between 0 and 1"
    else:
        print("Warning: 'rides_premium_lifetime' column not found. Setting Percent_rides_premium_lifetime to 0.")
        df['Percent_rides_premium_lifetime'] = 0
    
    return df

def create_plus_target(df):
    """Create binary target for Plus requests when it was NOT preselected."""
    # Keep all sessions - don't filter out preselected sessions
    # But only count Plus choices when Plus was NOT preselected
    df['chose_plus'] = ((df['requested_ride_type'] == 'plus') & (df['preselected_mode'] != 'plus')).astype(int)
    return df

def create_threshold_table(df, total_sessions_all, total_riders_all, thresholds=None, value_column='percent_rides_plus_lifetime', equality=False):
    """Create a table showing probability of choosing each ride type when it was NOT preselected, for sessions where value_column > threshold (or == threshold if equality=True)."""
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
            # Calculate rates where users chose each mode when it was NOT preselected
            plus_rate = subset['chose_plus'].mean()
            plus_sessions = subset['chose_plus'].sum()
            standard_saver_rate = ((subset['requested_ride_type'] == 'standard_saver') & (subset['preselected_mode'] != 'standard_saver')).mean()
            standard_saver_sessions = ((subset['requested_ride_type'] == 'standard_saver') & (subset['preselected_mode'] != 'standard_saver')).sum()
            premium_rate = ((subset['requested_ride_type'] == 'premium') & (subset['preselected_mode'] != 'premium')).mean()
            lux_rate = ((subset['requested_ride_type'] == 'lux') & (subset['preselected_mode'] != 'lux')).mean()
            fastpass_rate = ((subset['requested_ride_type'] == 'fastpass') & (subset['preselected_mode'] != 'fastpass')).mean()
            
            # Calculate rate where requested_ride_type matches preselected_mode
            chose_preselected_rate = (subset['requested_ride_type'] == subset['preselected_mode']).mean()
            
            total_sessions = len(subset)
            no_dominant_mode_mask = (subset['preselected_mode'] == 'standard') | (subset['preselected_mode'] == 'fastpass')
            distinct_riders_no_dominant_mode = subset['rider_lyft_id'][no_dominant_mode_mask].nunique()
            distinct_riders = distinct_riders_no_dominant_mode
            #distinct_riders = subset['rider_lyft_id'].nunique()

            sessions_pct = (total_sessions / total_sessions_all) * 100
            riders_pct = (distinct_riders / total_riders_all) * 100
            #riders_pct = (distinct_riders / total_riders_all) * 100 #exclude dominant mode preslection
            #subset['preselected_mode'] == 'standard' or subset['preselected_mode'] == 'plus'
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
                'fastpass_rate': fastpass_rate,
                'chose_preselected_rate': chose_preselected_rate,
                'chose_preselected_rate_pct': f"{chose_preselected_rate*100:.1f}%"
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
    premium_cnt_raw = df['rides_premium_lifetime'].sum()
    df['rides_premium_lifetime'] = df['rides_premium_lifetime'] - df['rides_premium_lifetime_pre_nov_2023']

    assert df['rides_premium_lifetime'].min() >= 0, "Rides premium lifetime should be non-negative"
    assert df['rides_premium_lifetime'].sum() != premium_cnt_raw, "Rides premium lifetime should be different from raw premium cnt"

    # Calculate totals for the whole dataset (for normalization)
    print("Calculating whole dataset totals for normalization...")
    df_whole = filter_by_segment(df, 'all')
    # Basic filtering for whole dataset
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df_whole = df_whole.dropna(subset=required_cols)
    df_whole = calculate_plus_percentage(df_whole)
    df_whole = calculate_premium_percentage(df_whole)
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
        df_segment = calculate_premium_percentage(df_segment)
        df_segment = create_plus_target(df_segment)
        # Analyze relationship
        # Remove analyze_plus_vs_plus_percentage and all calls to it
        # For percent_rides_plus_lifetime, call create_threshold_table directly and save the result

          # Calculate percent of riders with both percent_rides_plus_lifetime > 5% and percent_rides_premium_lifetime > 10%

        both_criteria = (
            (df_segment['percent_rides_plus_lifetime'] > 0.05) &
            (df_segment['Percent_rides_premium_lifetime'] > 0.10)
        )
        num_both = df_segment[both_criteria].rider_lyft_id.nunique()
        pct_both = num_both / total_riders_all * 100
        num_trigger_plus = df_segment[df_segment['percent_rides_plus_lifetime'] > 0.05].rider_lyft_id.nunique()
        pct_of_trigger_plus_trigger_premium = num_both / num_trigger_plus * 100


        print(segment_type)
        print(f"Riders with >5% Plus AND >10% Premium rides: {num_both:,} ({pct_both:.2f}%)")
        print(f"Riders with >5% Plus AND >10% Premium rides: {num_both:,} ({pct_both:.2f}%)")
        #print(f"% of Riders with >5% Plus also >10% Premium: {num_trigger_plus * 100:.2f}%")
        print(f"% of Riders with >5% Plus also >10% Premium: {pct_of_trigger_plus_trigger_premium}")

        # Create segment-specific output directories
        base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
        plots_dir = base_path / 'plots'
        reports_dir = base_path / f'reports/feature_threshold_analysis/{data_version}/segment_{segment_type}'
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
                
                # Create data-driven thresholds for better distribution
                # Use percentiles for most of the range, but include some logical breakpoints
                logical_thresholds = [0, 1, 2, 5, 10, 20]  # Common short distances
                
                # Add percentile-based thresholds for better data distribution
                # Cap at 95th percentile to avoid extreme outliers
                percentile_95 = np.percentile(dist_data, 95)
                
                if percentile_95 > 20:
                    # Create evenly spaced percentiles in the data-rich range
                    percentiles = [25, 50, 75, 80, 85, 90, 95]
                    percentile_thresholds = [np.percentile(dist_data, p) for p in percentiles]
                    
                    # Combine logical and percentile thresholds
                    all_thresholds = logical_thresholds + percentile_thresholds
                else:
                    all_thresholds = logical_thresholds
                
                # Remove duplicates, sort, and filter to reasonable range
                thresholds = sorted(set([t for t in all_thresholds if t <= percentile_95]))
                
                # Round thresholds to cleaner values for better readability
                rounded_thresholds = []
                for t in thresholds:
                    if t < 1:
                        rounded_thresholds.append(round(t, 1))
                    elif t < 10:
                        rounded_thresholds.append(round(t, 0))
                    elif t < 100:
                        rounded_thresholds.append(round(t / 5) * 5)  # Round to nearest 5
                    else:
                        rounded_thresholds.append(round(t / 10) * 10)  # Round to nearest 10
                
                thresholds = sorted(set(rounded_thresholds))
                
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

        # --- Cohort-specific analyses for Percent_rides_premium_lifetime ---
        if 'signup_year' in df_segment.columns:
            print(f"\n=== Cohort-specific Percent_rides_premium_lifetime analysis ===")
            
            # Cohort 1: signup_year <= 2020
            cohort_2020 = df_segment[df_segment['signup_year'] <= 2020].copy()
            if len(cohort_2020) >= 50:
                print(f"  Processing cohort (signup <= 2020): {len(cohort_2020)} sessions")
                cohort_2020_threshold_df = create_threshold_table(
                    cohort_2020,
                    value_column='Percent_rides_premium_lifetime',
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                cohort_2020_threshold_df.to_csv(reports_dir / f'threshold_by_Percent_rides_premium_lifetime_signup_2020_cohort.csv', index=False)
                print(f"  Saved signup ≤ 2020 cohort analysis")
            else:
                print(f"  Skipping signup ≤ 2020 cohort: only {len(cohort_2020)} sessions")
            
            # Cohort 2: signup_year <= 2021
            cohort_2021 = df_segment[df_segment['signup_year'] <= 2021].copy()
            if len(cohort_2021) >= 50:
                print(f"  Processing cohort (signup <= 2021): {len(cohort_2021)} sessions")
                cohort_2021_threshold_df = create_threshold_table(
                    cohort_2021,
                    value_column='Percent_rides_premium_lifetime',
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                cohort_2021_threshold_df.to_csv(reports_dir / f'threshold_by_Percent_rides_premium_lifetime_signup_2021_cohort.csv', index=False)
                print(f"  Saved signup ≤ 2021 cohort analysis")
            else:
                print(f"  Skipping signup ≤ 2021 cohort: only {len(cohort_2021)} sessions")
        else:
            print(f"  Skipping cohort analysis: 'signup_year' column not found")

        # --- Threshold-style table for lux_pin_eta_diff_wrt_standard_pin_eta_minutes ---
        if 'lux_pin_eta_diff_wrt_standard_pin_eta_minutes' in df_segment.columns:
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

        # --- Request rates by exact rides_plus_lifetime values (0-9) ---
        if 'rides_plus_lifetime' in df_segment.columns:
            print(f"\n=== Request rates by exact rides_plus_lifetime values (0-9) ===")
            plus_lifetime_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            
            # Filter to only include values that exist in the data
            available_values = []
            for val in plus_lifetime_values:
                if (df_segment['rides_plus_lifetime'] == val).any():
                    available_values.append(val)
            
            if available_values:
                threshold_df = create_threshold_table(
                    df_segment,
                    thresholds=available_values,
                    value_column='rides_plus_lifetime',
                    equality=True,
                    total_sessions_all=total_sessions_all,
                    total_riders_all=total_riders_all
                )
                threshold_df.to_csv(reports_dir / f'request_rates_by_rides_plus_lifetime_exact.csv', index=False)
                print(f"Saved request rates by exact rides_plus_lifetime values to: {reports_dir / f'request_rates_by_rides_plus_lifetime_exact.csv'}")
                print(f"Available rides_plus_lifetime values in data: {available_values}")
            else:
                print("No rides_plus_lifetime values in range 0-9 found in this segment.")
        else:
            print("Column 'rides_plus_lifetime' not found in data for this segment. Skipping exact rides_plus_lifetime analysis.")

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
    data_version = 'v4'
    main(segment_type_list, data_version) 