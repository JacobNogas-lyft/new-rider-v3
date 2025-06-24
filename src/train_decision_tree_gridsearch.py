import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, make_scorer, precision_score
from load_data import load_parquet_data
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def add_churned_indicator(df):
    """Add churned user indicator to the dataframe."""
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

def prepare_data(segment_type='all', mode='premium'):
    """Load and prepare data for training."""
    print(f"Loading and preparing data for segment: {segment_type}, mode: {mode}...")
    
    # Load data
    df = load_parquet_data()
    df = add_churned_indicator(df)
    
    # Filter by segment
    df = filter_by_segment(df, segment_type)
    
    # Create target variable
    df['target'] = ((df['requested_ride_type'] != df['preselected_mode']) & 
                   (df['requested_ride_type'] == mode)).astype(int)
    
    # Drop unnecessary columns
    cols_to_drop = [
        'requested_ride_type', 'preselected_mode', 'target',
        'purchase_session_id', 'candidate_product_key',
        'last_purchase_session_id', 'session_id', 'last_http_id', 'price_quote_id'
    ]
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    X = df[feature_cols]
    y = df['target']
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def create_precision_scorer():
    """Create a custom scorer that optimizes for precision of the positive class."""
    return make_scorer(precision_score, pos_label=1)

def perform_grid_search(X, y, segment_type, mode):
    """Perform grid search to find optimal hyperparameters."""
    print(f"\nPerforming GridSearchCV for {segment_type} segment, {mode} mode...")
    
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 8, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
        'class_weight': ['balanced']
    }
    
    # Create the model
    model = DecisionTreeClassifier(random_state=42)
    
    # Create custom scorer for precision
    precision_scorer = create_precision_scorer()
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=precision_scorer,
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=1,
        return_train_score=True
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation precision: {grid_search.best_score_:.4f}")
    
    return grid_search

def evaluate_best_model(grid_search, X, y, segment_type, mode):
    """Evaluate the best model on train/test split."""
    print(f"\nEvaluating best model for {segment_type} segment, {mode} mode...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Train on full training set
    best_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=[f'not {mode}', f'{mode} (not preselected)']))
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return best_model, X_test, y_test, y_pred, feature_importance

def save_results(grid_search, best_model, feature_importance, segment_type, mode):
    """Save results to files."""
    # Create output directory
    output_dir = Path(f'reports/gridsearch/{segment_type}/{mode}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grid search results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(output_dir / 'gridsearch_results.csv', index=False)
    
    # Save best parameters
    with open(output_dir / 'best_parameters.txt', 'w') as f:
        f.write(f"Best parameters: {grid_search.best_params_}\n")
        f.write(f"Best cross-validation precision: {grid_search.best_score_:.4f}\n")
        f.write(f"Best cross-validation score: {grid_search.best_score_:.4f}\n")
    
    # Save feature importance
    feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")

def create_visualizations(grid_search, feature_importance, segment_type, mode):
    """Create visualizations of the results."""
    # Create plots directory
    plots_dir = Path(f'plots/gridsearch/{segment_type}/{mode}')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 20 Feature Importances - {segment_type} segment, {mode} mode')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cross-validation scores heatmap
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Create a heatmap of max_depth vs min_samples_split
    pivot_data = cv_results.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_min_samples_split',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'CV Precision Scores - {segment_type} segment, {mode} mode\n(max_depth vs min_samples_split)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'cv_scores_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {plots_dir}")

def main(segment_type='all', mode='premium'):
    """Main function to run the complete pipeline."""
    print(f"Starting GridSearchCV optimization for {segment_type} segment, {mode} mode")
    print("="*60)
    
    # Prepare data
    X, y = prepare_data(segment_type, mode)
    
    # Check if we have enough positive samples
    positive_samples = y.sum()
    print(f"Positive samples: {positive_samples}")
    
    if positive_samples < 10:
        print("Warning: Very few positive samples. Results may be unreliable.")
    
    # Perform grid search
    grid_search = perform_grid_search(X, y, segment_type, mode)
    
    # Evaluate best model
    best_model, X_test, y_test, y_pred, feature_importance = evaluate_best_model(
        grid_search, X, y, segment_type, mode
    )
    
    # Save results
    save_results(grid_search, best_model, feature_importance, segment_type, mode)
    
    # Create visualizations
    create_visualizations(grid_search, feature_importance, segment_type, mode)
    
    print(f"\nGridSearchCV optimization completed for {segment_type} segment, {mode} mode!")
    return grid_search, best_model

if __name__ == "__main__":
    # Configuration
    segment_types = ['airport', 'churned', 'all']
    modes = ['premium']  # Can be extended to other modes
    
    # Run for each segment type
    for segment_type in segment_types:
        for mode in modes:
            try:
                grid_search, best_model = main(segment_type, mode)
                print(f"\n{'='*60}")
            except Exception as e:
                print(f"Error processing {segment_type} segment, {mode} mode: {e}")
                continue 