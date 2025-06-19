import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from load_data import load_parquet_data
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def add_churned_indicator(df):
    df['ds'] = pd.to_datetime(df['ds'])
    churned_mask = (
        (df['days_since_signup'] > 365) &
        (df['rides_lifetime'] > 2) &
        (df['all_type_total_rides_365d'] == 0)
    )
    df['is_churned_user'] = churned_mask.astype(int)
    return df

def load_and_prepare_data():
    print("Loading and preparing data...")
    df = load_parquet_data()
    df = add_churned_indicator(df)
    required_cols = ['requested_ride_type', 'preselected_mode']
    df = df.dropna(subset=required_cols)
    return df
    # Target: 1 if requested_ride_type != preselected_mode AND requested_ride_type == mode

def prepare_features_and_target(df, mode):
    df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == mode)).astype(int)
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    key_categorical_cols = ['region', 'currency', 'passenger_device', 'pax_os', 'pax_carrier']
    feature_cols = [col for col in numeric_cols if col not in drop_cols] + key_categorical_cols
    X = df[feature_cols]
    y = df['target_diff_mode']
    X = pd.get_dummies(X, drop_first=True)
    print("Data loaded and processed.")
    return X, y, feature_cols

def split_data(X, y):
    print("Splitting data into train and test sets (stratified)...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_decision_tree(X_train, y_train, max_depth):
    print(f"Training Decision Tree (max_depth={max_depth})...")
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    print("Model training complete.")
    return clf

def evaluate_model(clf, X_test, y_test, class_names, reports_dir):
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=class_names)
    reports_dir.mkdir(exist_ok=True, parents=True)
    report_path = reports_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(f"Saved classification report to {report_path}")

def visualize_tree(clf, X, class_names, plots_dir, max_depth):
    print("Visualizing tree...")
    plots_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(24, 8))
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, max_depth=min(3, max_depth), fontsize=8)
    plt.title(f'Decision Tree (first 3 levels, max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'decision_tree_plot.svg')
    print(f"Saved decision tree plot to {plots_dir / 'decision_tree_plot.svg'}")
    plt.figure(figsize=(24, 24))
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True, fontsize=8)
    plt.title(f'Decision Tree (full depth, max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'decision_tree_plot_full.svg')
    print(f"Saved full decision tree plot to {plots_dir / 'decision_tree_plot_full.svg'}")

def plot_feature_importance(clf, X, plots_dir):
    print("Plotting feature importances...")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 20
    plt.figure(figsize=(16, 8))
    plt.title('Feature Importances (Top 20)', fontsize=20)
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [X.columns[i] for i in indices[:top_n]], rotation=45, ha='right', fontsize=12)
    plt.ylabel('Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.pdf')
    print(f"Saved feature importance plot to {plots_dir / 'feature_importance.pdf'}")

def run_for_mode_and_depth(df, mode, max_depth):
    print(f"\n=== Running for mode: {mode}, max_depth: {max_depth} ===")
    X, y, feature_cols = prepare_features_and_target(df, mode)
    # If only one class, skip
    if y.nunique() < 2:
        print(f"Skipping mode={mode}, max_depth={max_depth}: only one class present.")
        return
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_decision_tree(X_train, y_train, max_depth)
    class_names = [f'not {mode}', f'{mode} (not preselected)']
    plots_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/decision_tree/mode_{mode}/max_depth_{max_depth}')
    reports_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/mode_{mode}/max_depth_{max_depth}')
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
    visualize_tree(clf, X, class_names, plots_dir, max_depth)
    plot_feature_importance(clf, X, plots_dir)

def main(mode_list, max_depth_list):
    df = load_and_prepare_data()
    # Parallelize using ProcessPoolExecutor
    tasks = []
    with ProcessPoolExecutor() as executor:
        for max_depth in max_depth_list:
            for mode in mode_list:
                tasks.append(executor.submit(run_for_mode_and_depth, df, mode, max_depth))
        for future in as_completed(tasks):
            future.result()  # To raise exceptions if any

if __name__ == "__main__":
    #main(mode_list=['fastpass', 'standard', 'premium', 'plus', 'lux', 'luxsuv'], max_depth_list=[3, 4, 5]) 
    main(mode_list=['standard_saver'], max_depth_list=[3, 4, 5]) 
    #TODO run this, as want to see if its really just standard_saver that is the potentiol