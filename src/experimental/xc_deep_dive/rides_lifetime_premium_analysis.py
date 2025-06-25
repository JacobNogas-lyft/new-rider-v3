import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add the src directory to the path so we can import load_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import load_parquet_data

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load the data and prepare for analysis."""
    print("Loading data...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def analyze_rides_lifetime_premium(df):
    """Analyze the rides_premium_lifetime column."""
    print("\n" + "="*60)
    print("RIDES PREMIUM LIFETIME ANALYSIS")
    print("="*60)
    
    # Check if column exists
    if 'rides_premium_lifetime' not in df.columns:
        print("ERROR: 'rides_premium_lifetime' column not found in dataset!")
        print("Available columns containing 'premium':")
        premium_cols = [col for col in df.columns if 'premium' in col.lower()]
        for col in premium_cols:
            print(f"  - {col}")
        return
    
    # Basic statistics
    premium_rides = df['rides_premium_lifetime']
    print(f"Column: rides_premium_lifetime")
    print(f"Data type: {premium_rides.dtype}")
    print(f"Total rows: {len(premium_rides):,}")
    print(f"Non-null values: {premium_rides.count():,}")
    print(f"Null values: {premium_rides.isnull().sum():,}")
    print(f"Null percentage: {(premium_rides.isnull().sum() / len(premium_rides)) * 100:.2f}%")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(f"Mean: {premium_rides.mean():.2f}")
    print(f"Median: {premium_rides.median():.2f}")
    print(f"Std: {premium_rides.std():.2f}")
    print(f"Min: {premium_rides.min()}")
    print(f"Max: {premium_rides.max()}")
    print(f"25th percentile: {premium_rides.quantile(0.25):.2f}")
    print(f"75th percentile: {premium_rides.quantile(0.75):.2f}")
    
    # Value distribution
    print(f"\nValue Distribution:")
    print(f"Zero values: {(premium_rides == 0).sum():,} ({(premium_rides == 0).sum() / len(premium_rides) * 100:.2f}%)")
    print(f"Values > 0: {(premium_rides > 0).sum():,} ({(premium_rides > 0).sum() / len(premium_rides) * 100:.2f}%)")
    print(f"Values >= 1: {(premium_rides >= 1).sum():,} ({(premium_rides >= 1).sum() / len(premium_rides) * 100:.2f}%)")
    print(f"Values >= 5: {(premium_rides >= 5).sum():,} ({(premium_rides >= 5).sum() / len(premium_rides) * 100:.2f}%)")
    print(f"Values >= 10: {(premium_rides >= 10).sum():,} ({(premium_rides >= 10).sum() / len(premium_rides) * 100:.2f}%)")
    
    return premium_rides

def analyze_target_correlation(df, premium_rides):
    """Analyze correlation between target_diff_mode and rides_premium_lifetime."""
    print("\n" + "="*60)
    print("TARGET CORRELATION ANALYSIS")
    print("="*60)
    
    # Create target variable
    df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == 'premium')).astype(int)
    
    # Basic target statistics
    target = df['target_diff_mode']
    print(f"Target variable: target_diff_mode")
    print(f"Target mean: {target.mean():.4f}")
    print(f"Target std: {target.std():.4f}")
    print(f"Target distribution:")
    print(f"  0 (not premium diff): {target.value_counts()[0]:,} ({target.value_counts()[0]/len(target)*100:.2f}%)")
    print(f"  1 (premium diff): {target.value_counts()[1]:,} ({target.value_counts()[1]/len(target)*100:.2f}%)")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    correlation = df['rides_premium_lifetime'].corr(df['target_diff_mode'])
    print(f"Pearson correlation: {correlation:.4f}")
    
    # Spearman correlation (for non-linear relationships)
    spearman_corr = df['rides_premium_lifetime'].corr(df['target_diff_mode'], method='spearman')
    print(f"Spearman correlation: {spearman_corr:.4f}")
    
    # Group analysis
    print(f"\nGroup Analysis:")
    target_0_mean = df[df['target_diff_mode'] == 0]['rides_premium_lifetime'].mean()
    target_1_mean = df[df['target_diff_mode'] == 1]['rides_premium_lifetime'].mean()
    print(f"Mean rides_premium_lifetime when target=0: {target_0_mean:.4f}")
    print(f"Mean rides_premium_lifetime when target=1: {target_1_mean:.4f}")
    print(f"Difference: {target_1_mean - target_0_mean:.4f}")
    print(f"Ratio: {target_1_mean / target_0_mean:.4f}")
    
    return df, target

def train_regression_models(df, premium_rides, target):
    """Train regression models to predict target_diff_mode from rides_premium_lifetime."""
    print("\n" + "="*60)
    print("REGRESSION MODEL TRAINING")
    print("="*60)
    
    # Filter for rows where rides_premium_lifetime > 0
    print(f"Filtering for rides_premium_lifetime > 0...")
    print(f"Original dataset size: {len(df):,}")
    
    filtered_df = df[df['rides_premium_lifetime'] > 0].copy()
    print(f"Filtered dataset size: {len(filtered_df):,}")
    print(f"Rows removed: {len(df) - len(filtered_df):,} ({(len(df) - len(filtered_df))/len(df)*100:.2f}%)")
    
    if len(filtered_df) == 0:
        print("No rows with rides_premium_lifetime > 0 found!")
        return None, None, None
    
    # Prepare features and target for filtered data
    X = filtered_df[['rides_premium_lifetime']].values
    y = filtered_df['target_diff_mode'].values
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution in filtered data:")
    print(f"  0: {np.sum(y == 0):,} ({np.sum(y == 0)/len(y)*100:.2f}%)")
    print(f"  1: {np.sum(y == 1):,} ({np.sum(y == 1)/len(y)*100:.2f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression (for continuous prediction)
    print(f"\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_lr_binary = (y_pred_lr > 0.5).astype(int)
    
    # Linear Regression metrics
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    accuracy_lr = accuracy_score(y_test, y_pred_lr_binary)
    
    print(f"Linear Regression Results:")
    print(f"  MSE: {mse_lr:.4f}")
    print(f"  R²: {r2_lr:.4f}")
    print(f"  Accuracy (threshold 0.5): {accuracy_lr:.4f}")
    print(f"  Coefficient: {lr_model.coef_[0]:.4f}")
    print(f"  Intercept: {lr_model.intercept_:.4f}")
    
    # Train Logistic Regression (for binary classification)
    print(f"\nTraining Logistic Regression...")
    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_log = log_model.predict(X_test_scaled)
    y_pred_log_proba = log_model.predict_proba(X_test_scaled)[:, 1]
    
    # Logistic Regression metrics
    accuracy_log = accuracy_score(y_test, y_pred_log)
    
    print(f"Logistic Regression Results:")
    print(f"  Accuracy: {accuracy_log:.4f}")
    print(f"  Coefficient: {log_model.coef_[0][0]:.4f}")
    print(f"  Intercept: {log_model.intercept_[0]:.4f}")
    
    # Classification report
    print(f"\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_log, target_names=['Not Premium Diff', 'Premium Diff']))
    
    return {
        'linear_model': lr_model,
        'logistic_model': log_model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_lr': y_pred_lr,
        'y_pred_log': y_pred_log,
        'y_pred_log_proba': y_pred_log_proba,
        'metrics': {
            'linear_mse': mse_lr,
            'linear_r2': r2_lr,
            'linear_accuracy': accuracy_lr,
            'logistic_accuracy': accuracy_log
        }
    }, filtered_df, X_train_scaled

def plot_histogram(premium_rides, save_path=None):
    """Plot histogram of rides_premium_lifetime."""
    print("\nCreating histogram...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of rides_premium_lifetime', fontsize=16, fontweight='bold')
    
    # 1. Full range histogram
    axes[0, 0].hist(premium_rides, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Full Range Distribution')
    axes[0, 0].set_xlabel('rides_premium_lifetime')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Zoomed in histogram (0-20 range)
    axes[0, 1].hist(premium_rides[premium_rides <= 20], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribution (0-20 range)')
    axes[0, 1].set_xlabel('rides_premium_lifetime')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Log scale histogram (excluding zeros)
    non_zero = premium_rides[premium_rides > 0]
    if len(non_zero) > 0:
        axes[1, 0].hist(non_zero, bins=30, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].set_title('Non-Zero Values Distribution')
        axes[1, 0].set_xlabel('rides_premium_lifetime')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No non-zero values', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Non-Zero Values Distribution')
    
    # 4. Box plot
    axes[1, 1].boxplot(premium_rides, vert=True)
    axes[1, 1].set_title('Box Plot')
    axes[1, 1].set_ylabel('rides_premium_lifetime')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    plt.show()

def plot_target_correlation(df, premium_rides, target, save_path=None):
    """Plot correlation analysis between target and rides_premium_lifetime."""
    print("\nCreating target correlation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Target vs rides_premium_lifetime Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    axes[0, 0].scatter(premium_rides, target, alpha=0.1, s=1)
    axes[0, 0].set_xlabel('rides_premium_lifetime')
    axes[0, 0].set_ylabel('target_diff_mode')
    axes[0, 0].set_title('Scatter Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot by target
    target_0_data = df[df['target_diff_mode'] == 0]['rides_premium_lifetime']
    target_1_data = df[df['target_diff_mode'] == 1]['rides_premium_lifetime']
    axes[0, 1].boxplot([target_0_data, target_1_data], labels=['target=0', 'target=1'])
    axes[0, 1].set_ylabel('rides_premium_lifetime')
    axes[0, 1].set_title('Distribution by Target')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram by target
    axes[1, 0].hist(target_0_data, bins=30, alpha=0.7, label='target=0', color='lightblue')
    axes[1, 0].hist(target_1_data, bins=30, alpha=0.7, label='target=1', color='lightcoral')
    axes[1, 0].set_xlabel('rides_premium_lifetime')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram by Target')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Probability of target=1 by rides_premium_lifetime bins
    # Create bins for rides_premium_lifetime
    df['premium_bins'] = pd.cut(df['rides_premium_lifetime'], bins=10, include_lowest=True)
    prob_by_bin = df.groupby('premium_bins')['target_diff_mode'].mean()
    
    axes[1, 1].bar(range(len(prob_by_bin)), prob_by_bin.values, color='gold')
    axes[1, 1].set_xlabel('rides_premium_lifetime Bins')
    axes[1, 1].set_ylabel('Probability of target=1')
    axes[1, 1].set_title('Probability of Premium Diff by Premium Rides Bins')
    axes[1, 1].set_xticks(range(len(prob_by_bin)))
    axes[1, 1].set_xticklabels([f"{interval.left:.1f}-{interval.right:.1f}" for interval in prob_by_bin.index], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        correlation_save_path = str(save_path).replace('.png', '_correlation.png')
        plt.savefig(correlation_save_path, dpi=300, bbox_inches='tight')
        print(f"Target correlation plots saved to: {correlation_save_path}")
    
    plt.show()

def plot_target_vs_premium_rides(df, save_path=None):
    """Plot average rides_premium_lifetime by target_diff_mode."""
    print("\nCreating target vs premium rides bar plot...")
    
    # Create target variable if not already present
    if 'target_diff_mode' not in df.columns:
        df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == 'premium')).astype(int)
    
    # Calculate averages
    avg_premium_rides = df.groupby('target_diff_mode')['rides_premium_lifetime'].mean()
    
    print(f"Average rides_premium_lifetime by target_diff_mode:")
    print(f"  Target=0 (Didn't Select Premium Or Selected Premium When Preselected): {avg_premium_rides[0]:.4f}")
    print(f"  Target=1 (Selected Premium When Not Preselected): {avg_premium_rides[1]:.4f}")
    print(f"  Difference: {avg_premium_rides[1] - avg_premium_rides[0]:.4f}")
    print(f"  Ratio: {avg_premium_rides[1] / avg_premium_rides[0]:.4f}")
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(['Target=0\n(Didn\'t Select Premium Or\nSelected Premium When Preselected)', 
                    'Target=1\n(Selected Premium When\nNot Preselected)'], 
                   avg_premium_rides.values, 
                   color=['lightblue', 'lightcoral'],
                   alpha=0.7,
                   edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_premium_rides.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average rides_premium_lifetime by Target', fontsize=14, fontweight='bold')
    plt.ylabel('Average rides_premium_lifetime', fontsize=12)
    plt.xlabel('Target (target_diff_mode)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    plt.text(0.02, 0.98, f'Difference: {avg_premium_rides[1] - avg_premium_rides[0]:.4f}\nRatio: {avg_premium_rides[1] / avg_premium_rides[0]:.2f}x', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        bar_save_path = str(save_path).replace('.png', '_target_vs_premium.png')
        plt.savefig(bar_save_path, dpi=300, bbox_inches='tight')
        print(f"Target vs premium rides bar plot saved to: {bar_save_path}")
    
    plt.show()

def plot_regression_results(models_results, filtered_df, save_path=None):
    """Plot regression analysis results."""
    if models_results is None:
        print("No regression results to plot.")
        return
    
    print("\nCreating regression analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Regression Analysis: rides_premium_lifetime vs target_diff_mode (Filtered: > 0)', fontsize=16, fontweight='bold')
    
    # Extract data
    X_test = models_results['X_test']
    y_test = models_results['y_test']
    y_pred_lr = models_results['y_pred_lr']
    y_pred_log = models_results['y_pred_log']
    y_pred_log_proba = models_results['y_pred_log_proba']
    
    # 1. Scatter plot with linear regression line
    axes[0, 0].scatter(X_test, y_test, alpha=0.5, s=10, color='blue', label='Actual')
    axes[0, 0].scatter(X_test, y_pred_lr, alpha=0.5, s=10, color='red', label='Linear Predicted')
    axes[0, 0].set_xlabel('rides_premium_lifetime')
    axes[0, 0].set_ylabel('target_diff_mode')
    axes[0, 0].set_title('Linear Regression: Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Logistic regression probability
    axes[0, 1].scatter(X_test, y_test, alpha=0.5, s=10, color='blue', label='Actual')
    axes[0, 1].scatter(X_test, y_pred_log_proba, alpha=0.5, s=10, color='green', label='Logistic Probability')
    axes[0, 1].set_xlabel('rides_premium_lifetime')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title('Logistic Regression: Probability')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals plot
    residuals = y_test - y_pred_lr
    axes[0, 2].scatter(y_pred_lr, residuals, alpha=0.5, s=10)
    axes[0, 2].axhline(y=0, color='red', linestyle='--')
    axes[0, 2].set_xlabel('Predicted Values')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residuals Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribution of rides_premium_lifetime by target
    target_0_data = filtered_df[filtered_df['target_diff_mode'] == 0]['rides_premium_lifetime']
    target_1_data = filtered_df[filtered_df['target_diff_mode'] == 1]['rides_premium_lifetime']
    
    axes[1, 0].hist(target_0_data, bins=30, alpha=0.7, label='target=0', color='lightblue', density=True)
    axes[1, 0].hist(target_1_data, bins=30, alpha=0.7, label='target=1', color='lightcoral', density=True)
    axes[1, 0].set_xlabel('rides_premium_lifetime')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution by Target (Filtered)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Model comparison
    models = ['Linear', 'Logistic']
    accuracies = [models_results['metrics']['linear_accuracy'], models_results['metrics']['logistic_accuracy']]
    axes[1, 1].bar(models, accuracies, color=['lightblue', 'lightgreen'])
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Model Accuracy Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 6. ROC-like curve (probability vs actual)
    # Sort by probability
    sorted_indices = np.argsort(y_pred_log_proba)
    sorted_proba = y_pred_log_proba[sorted_indices]
    sorted_actual = y_test[sorted_indices]
    
    # Calculate cumulative probabilities
    cumulative_actual = np.cumsum(sorted_actual) / np.sum(sorted_actual)
    cumulative_pred = np.cumsum(sorted_proba) / np.sum(sorted_proba)
    
    axes[1, 2].plot(cumulative_pred, cumulative_actual, linewidth=2, color='purple')
    axes[1, 2].plot([0, 1], [0, 1], '--', color='red', alpha=0.7)
    axes[1, 2].set_xlabel('Cumulative Predicted Probability')
    axes[1, 2].set_ylabel('Cumulative Actual')
    axes[1, 2].set_title('Probability Calibration')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        regression_save_path = str(save_path).replace('.png', '_regression.png')
        plt.savefig(regression_save_path, dpi=300, bbox_inches='tight')
        print(f"Regression analysis plots saved to: {regression_save_path}")
    
    plt.show()

def plot_additional_analysis(df, premium_rides, save_path=None):
    """Additional analysis plots."""
    print("\nCreating additional analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Additional Analysis of rides_premium_lifetime', fontsize=16, fontweight='bold')
    
    # 1. Cumulative distribution
    sorted_values = np.sort(premium_rides)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    axes[0, 0].plot(sorted_values, cumulative, linewidth=2)
    axes[0, 0].set_title('Cumulative Distribution')
    axes[0, 0].set_xlabel('rides_premium_lifetime')
    axes[0, 0].set_ylabel('Cumulative Probability')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Relationship with requested_ride_type (if available)
    if 'requested_ride_type' in df.columns:
        premium_by_type = df.groupby('requested_ride_type')['rides_premium_lifetime'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(premium_by_type)), premium_by_type.values, color='lightcoral')
        axes[0, 1].set_title('Average rides_premium_lifetime by Requested Ride Type')
        axes[0, 1].set_xlabel('Requested Ride Type')
        axes[0, 1].set_ylabel('Average rides_premium_lifetime')
        axes[0, 1].set_xticks(range(len(premium_by_type)))
        axes[0, 1].set_xticklabels(premium_by_type.index, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Relationship with preselected_mode (if available)
    if 'preselected_mode' in df.columns:
        premium_by_preselected = df.groupby('preselected_mode')['rides_premium_lifetime'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(premium_by_preselected)), premium_by_preselected.values, color='lightblue')
        axes[1, 0].set_title('Average rides_premium_lifetime by Preselected Mode')
        axes[1, 0].set_xlabel('Preselected Mode')
        axes[1, 0].set_ylabel('Average rides_premium_lifetime')
        axes[1, 0].set_xticks(range(len(premium_by_preselected)))
        axes[1, 0].set_xticklabels(premium_by_preselected.index, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribution by region (if available)
    if 'region' in df.columns:
        # Get top 10 regions by count
        top_regions = df['region'].value_counts().head(10).index
        region_data = df[df['region'].isin(top_regions)]
        region_data.boxplot(column='rides_premium_lifetime', by='region', ax=axes[1, 1])
        axes[1, 1].set_title('rides_premium_lifetime by Region (Top 10)')
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('rides_premium_lifetime')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Save with different name to avoid overwriting
        additional_save_path = str(save_path).replace('.png', '_additional.png')
        plt.savefig(additional_save_path, dpi=300, bbox_inches='tight')
        print(f"Additional analysis plots saved to: {additional_save_path}")
    
    plt.show()

def plot_probability_by_thresholds(df, save_path=None):
    """Plot probability of selecting premium when not preselected for different rides_premium_lifetime thresholds."""
    print("\nCreating probability by thresholds plot...")
    
    # Create target variable if not already present
    if 'target_diff_mode' not in df.columns:
        df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == 'premium')).astype(int)
    
    # Define thresholds
    thresholds = [1, 2, 3, 4, 5]
    
    # Calculate probabilities for each threshold
    probabilities = []
    counts = []
    
    print(f"Probability of 'Selected Premium When Not Preselected' by rides_premium_lifetime threshold:")
    
    for threshold in thresholds:
        # Filter for users with rides_premium_lifetime >= threshold
        filtered_df = df[df['rides_premium_lifetime'] >= threshold]
        
        if len(filtered_df) > 0:
            # Calculate probability of target=1
            prob = filtered_df['target_diff_mode'].mean()
            count = len(filtered_df)
            
            probabilities.append(prob)
            counts.append(count)
            
            print(f"  Threshold >= {threshold}: {prob:.4f} ({count:,} users)")
        else:
            probabilities.append(0)
            counts.append(0)
            print(f"  Threshold >= {threshold}: No users found")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Probability
    bars1 = ax1.bar([str(t) for t in thresholds], probabilities, 
                    color='lightcoral', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, prob in zip(bars1, probabilities):
        if prob > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{prob:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Probability of Selecting Premium When Not Preselected', fontsize=14, fontweight='bold')
    ax1.set_xlabel('rides_premium_lifetime Threshold', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(probabilities) * 1.1 if max(probabilities) > 0 else 0.1)
    
    # Plot 2: Number of users
    bars2 = ax2.bar([str(t) for t in thresholds], counts, 
                    color='lightblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars2, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Number of Users by Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('rides_premium_lifetime Threshold', fontsize=12)
    ax2.set_ylabel('Number of Users', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        threshold_save_path = str(save_path).replace('.png', '_probability_thresholds.png')
        plt.savefig(threshold_save_path, dpi=300, bbox_inches='tight')
        print(f"Probability by thresholds plot saved to: {threshold_save_path}")
    
    plt.show()
    
    # Additional analysis: Compare with overall probability
    overall_prob = df['target_diff_mode'].mean()
    print(f"\nOverall probability (all users): {overall_prob:.4f}")
    print(f"Probability ratios compared to overall:")
    for i, threshold in enumerate(thresholds):
        if probabilities[i] > 0:
            ratio = probabilities[i] / overall_prob
            print(f"  Threshold >= {threshold}: {ratio:.2f}x higher than overall")

def main():
    """Main function to run the analysis."""
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Analyze rides_premium_lifetime
        premium_rides = analyze_rides_lifetime_premium(df)
        
        if premium_rides is not None:
            # Analyze target correlation
            df, target = analyze_target_correlation(df, premium_rides)
            
            # Create plots directory
            plots_dir = Path('/home/sagemaker-user/studio/src/new-rider-v3/plots/xc_deep_dive')
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            # Plot histogram
            histogram_path = plots_dir / 'rides_premium_lifetime_histogram.png'
            plot_histogram(premium_rides, save_path=histogram_path)
            
            # Plot target correlation
            plot_target_correlation(df, premium_rides, target, save_path=histogram_path)
            
            # Plot target vs premium rides bar plot
            plot_target_vs_premium_rides(df, save_path=histogram_path)
            
            # Plot probability by thresholds
            plot_probability_by_thresholds(df, save_path=histogram_path)
            
            # Train regression models (filtered for rides_premium_lifetime > 0)
            models_results, filtered_df, X_train_scaled = train_regression_models(df, premium_rides, target)
            
            # Plot regression results
            plot_regression_results(models_results, filtered_df, save_path=histogram_path)
            
            # Plot additional analysis
            plot_additional_analysis(df, premium_rides, save_path=histogram_path)
            
            print(f"\nAnalysis completed!")
            print(f"Plots saved to: {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 