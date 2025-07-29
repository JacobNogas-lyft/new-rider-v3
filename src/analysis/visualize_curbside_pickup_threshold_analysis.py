import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import warnings
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_csv_files(base_path):
    """
    Load all CSV files from the curbside_pickup_threshold_analysis folder structure.
    Returns a dictionary with segment names as keys and lists of (filename, dataframe) tuples as values.
    """
    base_path = Path(base_path)
    segments_data = {}
    
    # Find all segment folders
    segment_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('segment_')]
    
    for segment_folder in segment_folders:
        segment_name = segment_folder.name.replace('segment_', '')
        segments_data[segment_name] = []
        
        # Load all CSV files in this segment folder
        csv_files = list(segment_folder.glob('*.csv'))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                segments_data[segment_name].append((csv_file.stem, df))
                print(f"Loaded: {csv_file}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    return segments_data

def extract_feature_name(filename):
    """
    Extract feature name from filename.
    Examples:
    - 'threshold_by_percent_rides_plus_lifetime' -> 'percent_rides_plus_lifetime'
    - 'threshold_by_airline_pickup' -> 'airline_pickup'
    - 'descriptive_stats_lux_pin_eta_diff_minutes' -> 'lux_pin_eta_diff_minutes'
    """
    if 'threshold_by_' in filename:
        return filename.split('threshold_by_')[1]
    elif 'descriptive_stats_' in filename:
        return filename.split('descriptive_stats_')[1]
    else:
        return filename

def is_categorical_feature(feature_name):
    # Only treat these as categorical for bar charts
    categorical_features = ['airline_destination', 'airline_pickup', 'signup_year']
    return feature_name in categorical_features

def calculate_r2(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 2:
        return np.nan
    coeffs = np.polyfit(x_fit, y_fit, 1)
    y_pred = np.polyval(coeffs, x_fit)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2

def create_visualization(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/curbside_pickup_threshold_analysis'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the ride type columns and new labels
    ride_type_columns = ['plus_probability', 'standard_saver_rate', 'premium_rate', 'lux_rate', 'fastpass_rate']
    ride_type_labels = ['XL', 'WS', 'XC', 'Black', 'PP']
    label_map = dict(zip(ride_type_columns, ride_type_labels))
    
    # Colors for each ride type
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#000000']
    
    for segment_name, files_data in segments_data.items():
        print(f"\nCreating visualization for segment: {segment_name}")
        
        # Filter valid files for plotting (exclude descriptive stats files)
        valid_files = []
        for filename, df in files_data:
            # Skip descriptive statistics files
            if 'descriptive_stats_' in filename:
                continue
                
            feature_name = extract_feature_name(filename)
            if feature_name == 'years_since_signup':
                continue
            if segment_name == 'airport_pickup' and feature_name == 'airline_destination':
                continue
            if segment_name == 'airport_dropoff' and feature_name == 'airline_pickup':
                continue
            if segment_name not in ['airport_pickup', 'airport_dropoff'] and feature_name in ['airline_pickup', 'airline_destination']:
                continue
            valid_files.append((filename, df))
        
        n_files = len(valid_files)
        n_cols = 1
        n_rows = n_files
        if n_files == 0:
            continue
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6*n_rows))
        if n_files == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (filename, df) in enumerate(valid_files):
            ax = axes[idx]
            feature_name = extract_feature_name(filename)
            
            # Remove years_since_signup from all plots
            if feature_name == 'years_since_signup':
                fig.delaxes(ax)
                continue

            # Filtering logic for features
            if segment_name == 'airport_pickup' and feature_name == 'airline_destination':
                fig.delaxes(ax)
                continue
            if segment_name == 'airport_dropoff' and feature_name == 'airline_pickup':
                fig.delaxes(ax)
                continue
            if segment_name not in ['airport_pickup', 'airport_dropoff'] and feature_name in ['airline_pickup', 'airline_destination']:
                fig.delaxes(ax)
                continue

            # Determine x-axis column
            if 'threshold' in df.columns:
                x_col = 'threshold'
                x_label = 'Threshold'
            elif 'threshold_pct' in df.columns:
                x_col = 'threshold_pct'
                x_label = 'Threshold (%)'
            else:
                potential_x_cols = [col for col in df.columns if col not in ride_type_columns + ['total_sessions', 'sessions_pct', 'distinct_riders', 'riders_pct', 'plus_sessions', 'standard_saver_sessions']]
                x_col = potential_x_cols[0] if potential_x_cols else df.columns[0]
                x_label = x_col.replace('_', ' ').title()

            # Use bar chart only for known categorical features
            if is_categorical_feature(feature_name):
                x_categories = df[x_col].astype(str)
                bar_width = 0.13
                x = np.arange(len(x_categories))
                n_bars = 0
                for i, (col, label, color) in enumerate(zip(ride_type_columns, ride_type_labels, colors)):
                    if col in df.columns:
                        if df[col].dtype == 'object' and '%' in str(df[col].iloc[0]):
                            values = df[col].str.rstrip('%').astype(float) / 100
                        else:
                            values = df[col].astype(float)
                        if values.notna().any() and values.sum() != 0:
                            ax.bar(x + i*bar_width, values, width=bar_width, label=label, color=color, align='center')
                            n_bars += 1
                if n_bars > 0:
                    # Set x-tick labels to just the threshold/category values
                    if is_categorical_feature(feature_name) or df[x_col].dtype == 'object':
                        xvals = np.arange(len(df[x_col]))
                        ax.set_xticks(xvals)
                        ax.set_xticklabels([str(x) for x in df[x_col]], rotation=45, ha='right')
                        plt.subplots_adjust(bottom=0.2)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
            else:
                # Connected line plot for all other features
                for i, (col, label, color) in enumerate(zip(ride_type_columns, ride_type_labels, colors)):
                    if col in df.columns:
                        if df[col].dtype == 'object' and '%' in str(df[col].iloc[0]):
                            values = df[col].str.rstrip('%').astype(float)
                        else:
                            values = df[col].astype(float) * 100
                        if x_col == 'threshold_pct':
                            x_values = df[x_col]
                        else:
                            x_values = pd.to_numeric(df[x_col], errors='coerce')
                        # Calculate R^2 for the line
                        r2 = calculate_r2(np.array(x_values, dtype=float), np.array(values, dtype=float))
                        line = ax.plot(x_values, values, label=f"{label} R² = {r2:.2f}", color=color, linewidth=2, marker='o', markersize=4)
                        
                        # Add text annotations for each data point
                        for x, y in zip(x_values, values):
                            if not np.isnan(x) and not np.isnan(y):
                                # Format the value as percentage if it's small, otherwise as decimal
                                text_val = f"{y:.1f}%"
                                ax.annotate(text_val, (x, y), 
                                          xytext=(0, 5), textcoords='offset points',
                                          ha='center', va='bottom', fontsize=8, color=color)
                
                # Set x-tick labels to just the threshold/category values
                if is_categorical_feature(feature_name) or df[x_col].dtype == 'object':
                    xvals = np.arange(len(df[x_col]))
                    ax.set_xticks(xvals)
                    ax.set_xticklabels([str(x) for x in df[x_col]], rotation=45, ha='right')
                    plt.subplots_adjust(bottom=0.2)
                else:
                    xvals = np.array(df[x_col])
                    ax.set_xticks(xvals)
                    ax.set_xticklabels([str(x) for x in xvals], rotation=45, ha='right')
                    plt.subplots_adjust(bottom=0.2)
                if 'x_values' in locals() and len(x_values) > 10:
                    ax.tick_params(axis='x', rotation=45)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('Request Rate (%)')
            ax.set_title(f'{feature_name.replace("_", " ").title()} - Curbside Pickup (LAX/ORD/SFO)', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for idx in range(n_files, len(axes)):
            fig.delaxes(axes[idx])
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(right=0.85)  # Make room for legend
        
        # Save the plot
        output_file = output_path / f'curbside_pickup_threshold_analysis_{segment_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def create_summary_visualization(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/curbside_pickup_threshold_analysis'):
    """
    Create a summary visualization comparing segments for key features.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find common features across segments (excluding descriptive stats)
    common_features = set()
    for segment_name, files_data in segments_data.items():
        for filename, _ in files_data:
            if 'descriptive_stats_' not in filename:
                feature = extract_feature_name(filename)
                common_features.add(feature)
    
    # Focus on key features
    key_features = ['percent_rides_plus_lifetime', 'Percent_rides_premium_lifetime', 
                   'lux_pin_eta_diff_wrt_standard_pin_eta_minutes', 'lux_final_price_diff_wrt_standard_major_currency']
    available_features = [f for f in key_features if f in common_features]
    
    if not available_features:
        print("No common key features found for summary visualization")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    ride_type_columns = ['plus_probability', 'standard_saver_rate', 'premium_rate', 'lux_rate', 'fastpass_rate']
    ride_type_labels = ['XL', 'WS', 'XC', 'Black', 'PP']
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#000000']
    
    for idx, feature in enumerate(available_features[:4]):  # Max 4 subplots
        ax = axes[idx]
        
        for segment_name, files_data in segments_data.items():
            # Find the file for this feature
            feature_file = None
            for filename, df in files_data:
                if 'descriptive_stats_' not in filename and extract_feature_name(filename) == feature:
                    feature_file = (filename, df)
                    break
            
            if feature_file is None:
                continue
                
            filename, df = feature_file
            
            # Determine x-axis column
            if 'threshold' in df.columns:
                x_col = 'threshold'
            elif 'threshold_pct' in df.columns:
                x_col = 'threshold_pct'
            else:
                potential_x_cols = [col for col in df.columns if col not in ride_type_columns + ['total_sessions', 'sessions_pct', 'distinct_riders', 'riders_pct', 'plus_sessions', 'standard_saver_sessions']]
                x_col = potential_x_cols[0] if potential_x_cols else df.columns[0]
            
            # Plot Plus rate for this segment
            if 'plus_probability' in df.columns:
                if df['plus_probability'].dtype == 'object' and '%' in str(df['plus_probability'].iloc[0]):
                    values = df['plus_probability'].str.rstrip('%').astype(float) / 100
                else:
                    values = df['plus_probability'].astype(float)
                
                if x_col == 'threshold_pct':
                    x_values = df[x_col]
                else:
                    x_values = pd.to_numeric(df[x_col], errors='coerce')
                
                line = ax.plot(x_values, values, marker='o', label=f'{segment_name} (XL)', 
                       linewidth=2, markersize=4, alpha=0.8)
                
                # Add text annotations for each data point
                for x, y in zip(x_values, values):
                    if not np.isnan(x) and not np.isnan(y):
                        text_val = f"{y*100:.1f}%"
                        ax.annotate(text_val, (x, y), 
                                  xytext=(0, 5), textcoords='offset points',
                                  ha='center', va='bottom', fontsize=8, color='black')
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Plus Rate')
        ax.set_title(f'Plus Rate by {feature.replace("_", " ").title()} - Curbside Pickup (LAX/ORD/SFO)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if len(x_values) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(available_features), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the summary plot
    output_file = output_path / 'curbside_pickup_threshold_analysis_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary: {output_file}")
    plt.close()

def create_riders_pct_bar_plots(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/curbside_pickup_threshold_analysis_riders_pct'):
    """
    Create bar plots showing the percentage of riders for each threshold.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_col = 'threshold'

    for segment_name, files_data in segments_data.items():
        # Filter valid files for plotting (exclude descriptive stats files)
        valid_files = []
        for filename, df in files_data:
            # Skip descriptive statistics files
            if 'descriptive_stats_' in filename:
                continue
                
            feature_name = extract_feature_name(filename)
            if feature_name == 'years_since_signup':
                continue
            if segment_name == 'airport_pickup' and feature_name == 'airline_destination':
                continue
            if segment_name == 'airport_dropoff' and feature_name == 'airline_pickup':
                continue
            if segment_name not in ['airport_pickup', 'airport_dropoff'] and feature_name in ['airline_pickup', 'airline_destination']:
                continue
            if 'riders_pct' not in df.columns:
                continue
            valid_files.append((filename, df))
        
        n_files = len(valid_files)
        n_cols = 1
        n_rows = n_files
        if n_files == 0:
            continue
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6*n_rows))
        if n_files == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (filename, df) in enumerate(valid_files):
            ax = axes[idx]
            feature_name = extract_feature_name(filename)
            
            # Remove years_since_signup from all plots
            if feature_name == 'years_since_signup':
                ax.set_visible(False)
                continue
            # Filtering logic for features
            if segment_name == 'airport_pickup' and feature_name == 'airline_destination':
                ax.set_visible(False)
                continue
            if segment_name == 'airport_dropoff' and feature_name == 'airline_pickup':
                ax.set_visible(False)
                continue
            if segment_name not in ['airport_pickup', 'airport_dropoff'] and feature_name in ['airline_pickup', 'airline_destination']:
                ax.set_visible(False)
                continue
            if 'riders_pct' not in df.columns:
                ax.set_visible(False)
                continue
            
            yvals = df['riders_pct']
            # Convert to float if needed, then to percent
            if yvals.dtype == 'object':
                yvals = yvals.str.rstrip('%').astype(float)
            else:
                yvals = yvals.astype(float)
            
            # Always use index for bar chart xvals
            xvals = np.arange(len(df[x_col]))
            xlabels = [str(x) for x in df[x_col]]
            ax.set_xticks(xvals)
            ax.set_xticklabels(xlabels, rotation=45, ha='right')
            bars = ax.bar(xvals, yvals, color='tab:blue', alpha=0.8)
            
            # Add text values above bars as percent
            for bar, val in zip(bars, yvals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=9, color='black', rotation=0)
            
            # Set x-axis label to the feature name, not 'threshold'
            ax.set_xlabel(feature_name.replace('_', ' ').title())
            ax.set_ylabel('% of All Riders')
            ax.set_title(f'% of All Riders vs {feature_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / f'curbside_pickup_riders_pct_{segment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_descriptive_stats_visualization(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/curbside_pickup_threshold_analysis'):
    """
    Create visualizations for descriptive statistics of lux features.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Features to visualize stats for
    lux_features = ['lux_pin_eta_diff_minutes', 'lux_final_price_diff_currency']
    
    for segment_name, files_data in segments_data.items():
        stats_files = []
        special_stats_file = None
        
        for filename, df in files_data:
            if 'descriptive_stats_' in filename:
                feature_name = extract_feature_name(filename)
                if any(lux_feat in feature_name for lux_feat in lux_features):
                    stats_files.append((filename, df, feature_name))
            elif 'special_rider_statistics' in filename:
                special_stats_file = (filename, df)
        
        if not stats_files:
            continue
        
        # Create subplot layout: descriptive stats + special rider statistic
        n_subplots = len(stats_files) + (1 if special_stats_file else 0)
        fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 6*n_subplots))
        if n_subplots == 1:
            axes = [axes]
        
        # Plot descriptive statistics
        for idx, (filename, df, feature_name) in enumerate(stats_files):
            ax = axes[idx]
            
            # Extract key statistics for plotting
            stats_to_plot = ['mean', 'median', 'std', 'q25', 'q75', 'q90', 'q95']
            values = []
            labels = []
            
            for stat in stats_to_plot:
                if stat in df.columns:
                    values.append(df[stat].iloc[0])
                    labels.append(stat.upper())
            
            if values:
                bars = ax.bar(labels, values, color='skyblue', alpha=0.8)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}',
                            ha='center', va='bottom', fontsize=10, color='black')
                
                ax.set_title(f'Descriptive Statistics: {feature_name.replace("_", " ").title()}', 
                           fontsize=12, fontweight='bold')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Plot special rider statistic if available
        if special_stats_file:
            ax = axes[-1]
            filename, df = special_stats_file
            
            if 'value' in df.columns and len(df) > 0:
                # Define metric labels and colors (includes both old and new metric names for compatibility)
                metric_labels = {
                    # New SFO-inclusive metric names
                    'pct_riders_lax_ord_sfo': 'LAX/ORD/SFO\nSessions\n(% of all riders)',
                    'pct_sessions_lax_ord_sfo': 'LAX/ORD/SFO\nSessions\n(% of all sessions)',
                    'pct_lax_ord_sfo_riders_with_black_atf': 'Black ATF\n(% of LAX/ORD/SFO\nriders)',
                    'pct_riders_lax_ord_sfo_and_lux_price_lte_44': 'LAX/ORD/SFO +\nLux ≤ $44\n(% of all riders)',
                    'pct_sessions_lax_ord_sfo_and_lux_price_lte_44': 'LAX/ORD/SFO +\nLux ≤ $44\n(% of all sessions)',
                    # Old LAX/ORD-only metric names for backward compatibility
                    'pct_riders_lax_ord': 'LAX/ORD\nSessions\n(% of all riders)',
                    'pct_sessions_lax_ord': 'LAX/ORD\nSessions\n(% of all sessions)',
                    'pct_lax_ord_riders_with_black_atf': 'Black ATF\n(% of LAX/ORD\nriders)',
                    'pct_riders_lax_ord_and_lux_price_lte_44': 'LAX/ORD +\nLux ≤ $44\n(% of all riders)',
                    'pct_sessions_lax_ord_and_lux_price_lte_44': 'LAX/ORD +\nLux ≤ $44\n(% of all sessions)',
                    # Common metric
                    'pct_sessions_black_atf': 'Black ATF\n(% of all\nsessions)'
                }
                colors = ['skyblue', 'lightsteelblue', 'lightgreen', 'orange', 'mediumorchid', 'coral']
                
                # Extract values for each metric
                values = []
                labels = []
                bar_colors = []
                
                for idx, (_, row) in enumerate(df.iterrows()):
                    metric = row['metric']
                    value = row['value']
                    if metric in metric_labels:
                        values.append(value)
                        labels.append(metric_labels[metric])
                        bar_colors.append(colors[idx % len(colors)])
                
                if values:
                    # Create bars for all special statistics
                    bars = ax.bar(labels, values, color=bar_colors, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}%',
                                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
                    
                    ax.set_title('LAX/ORD/SFO Summary', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Percentage (%)')
                    ax.grid(True, alpha=0.3, axis='y')
                    # Set y-axis limit based on maximum value
                    max_val = max(values) if values else 1.0
                    ax.set_ylim(0, max(max_val * 1.2, 1.0))
                    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path / f'curbside_pickup_descriptive_stats_{segment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved descriptive stats visualization for {segment_name}")

def main():
    """
    Main function to run the curbside pickup visualization (LAX/ORD/SFO).
    """
    print("Loading curbside pickup threshold analysis data (LAX/ORD/SFO)...")
    
    # Load data from the curbside pickup reports directory
    base_path = "/home/sagemaker-user/studio/src/new-rider-v3/reports/curbside_pickup_threshold_analysis"
    print(f"Looking for data in: {base_path}")
    
    # Check if directory exists
    if not Path(base_path).exists():
        print(f"Directory does not exist: {base_path}")
        print("Current working directory:", os.getcwd())
        print("Available directories in reports:")
        reports_dir = Path("/home/sagemaker-user/studio/src/new-rider-v3/reports")
        if reports_dir.exists():
            for item in reports_dir.iterdir():
                print(f"  - {item.name}")
        return
    
    segments_data = load_csv_files(base_path)
    
    if not segments_data:
        print("No data found. Please check the path: reports/curbside_pickup_threshold_analysis")
        return
    
    print(f"\nFound {len(segments_data)} segments:")
    for segment_name, files_data in segments_data.items():
        print(f"  - {segment_name}: {len(files_data)} files")
    
    # Create visualizations
    print("\nCreating curbside pickup visualizations...")
    create_visualization(segments_data)
    create_summary_visualization(segments_data)
    create_riders_pct_bar_plots(segments_data)
    create_descriptive_stats_visualization(segments_data)
    
    print("\nCurbside pickup visualization complete (LAX/ORD/SFO)!")
    print("Check the following directories:")
    print("  - plots/curbside_pickup_threshold_analysis/")
    print("  - plots/curbside_pickup_threshold_analysis_riders_pct/")

if __name__ == "__main__":
    main() 