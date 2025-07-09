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
    Load all CSV files from the feature_threshold_analysis folder structure.
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
    """
    if 'threshold_by_' in filename:
        return filename.split('threshold_by_')[1]
    else:
        return filename

def is_categorical_feature(feature_name):
    # Only treat these as categorical for bar charts
    categorical_features = ['airline_destination', 'airline_pickup']
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

def format_xtick_labels_with_riders_pct(x_labels, riders_pct):
    # x_labels: list/array of x values (categories or thresholds)
    # riders_pct: list/array of percentages (as strings or floats)
    formatted = []
    for label, pct in zip(x_labels, riders_pct):
        # Try to extract numeric value from pct if it's a string with %
        if isinstance(pct, str) and '%' in pct:
            pct_val = pct.strip()
        else:
            try:
                pct_val = f"{float(pct):.1f}%"
            except:
                pct_val = str(pct)
        formatted.append(f"{label}\n({pct_val})")
    return formatted

def create_visualization(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/feature_threshold_analysis'):
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
        
        # Filter valid files for plotting
        valid_files = []
        for filename, df in files_data:
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
            
            # if feature_name == 'percent_rides_plus_lifetime':
            #     x_label = '% Lifetime XL Rides > x'

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
                    # Set x-tick labels to just the threshold/category values (no riders_pct)
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
                # Set x-tick labels to just the threshold/category values (no riders_pct)
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
            ax.set_title(f'{feature_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for idx in range(n_files, len(axes)):
            fig.delaxes(axes[idx])
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(right=0.85)  # Make room for legend
        
        # Save the plot
        output_file = output_path / f'feature_threshold_analysis_{segment_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def create_summary_visualization(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/feature_threshold_analysis'):
    """
    Create a summary visualization comparing segments for key features.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find common features across segments
    common_features = set()
    for segment_name, files_data in segments_data.items():
        for filename, _ in files_data:
            feature = extract_feature_name(filename)
            common_features.add(feature)
    
    # Focus on key features
    key_features = ['percent_rides_plus_lifetime', 'years_since_signup', 'airline_destination', 'airline_pickup']
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
                if extract_feature_name(filename) == feature:
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
        ax.set_title(f'Plus Rate by {feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if len(x_values) > 10:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(len(available_features), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the summary plot
    output_file = output_path / 'feature_threshold_analysis_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary: {output_file}")
    plt.close()

def create_riders_pct_bar_plots(segments_data, output_dir='/home/sagemaker-user/studio/src/new-rider-v3/plots/feature_threshold_analysis_riders_pct'):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_col = 'threshold'

    for segment_name, files_data in segments_data.items():
        # Filter valid files for plotting
        valid_files = []
        for filename, df in files_data:
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
            # If values are in 0-1, convert to 0-100
            # if yvals.max() <= 1.0:
            #     yvals = yvals * 100
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
            ax.set_ylabel('% of New Riders')
            ax.set_title(f'% of New Riders vs {feature_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path / f'riders_pct_{segment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to run the visualization.
    """
    print("Loading feature threshold analysis data...")
    
    # Load data from the reports directory - use absolute path
    base_path = "/home/sagemaker-user/studio/src/new-rider-v3/reports/feature_threshold_analysis"
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
        print("No data found. Please check the path: reports/feature_threshold_analysis")
        return
    
    print(f"\nFound {len(segments_data)} segments:")
    for segment_name, files_data in segments_data.items():
        print(f"  - {segment_name}: {len(files_data)} files")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualization(segments_data)
    create_summary_visualization(segments_data)
    create_riders_pct_bar_plots(segments_data)
    
    print("\nVisualization complete! Check the '/home/sagemaker-user/studio/src/new-rider-v3/plots/feature_threshold_analysis' and 'plots/feature_threshold_analysis_riders_pct' directories.")

if __name__ == "__main__":
    main() 