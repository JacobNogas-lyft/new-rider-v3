import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from utils.load_data import load_parquet_data
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

def prepare_features_and_target(df, mode):
    df = df.copy()
    df['target_diff_mode'] = ((df['requested_ride_type'] != df['preselected_mode']) & (df['requested_ride_type'] == mode)).astype(int)
    drop_cols = ['target_diff_mode', 'requested_ride_type', 'preselected_mode']
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    key_categorical_cols = ['region', 'currency', 'passenger_device', 'pax_os', 'pax_carrier']
    feature_cols = [col for col in numeric_cols if col not in drop_cols] + key_categorical_cols
    X = df[feature_cols]
    y = df['target_diff_mode']
    X = pd.get_dummies(X, drop_first=True)
    return X, y, feature_cols

def split_data(X, y):
    print("Splitting data into train and test sets (stratified)...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_xgboost(X_train, y_train, max_depth):
    print(f"Training XGBoost model (max_depth={max_depth})...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    clf = XGBClassifier(max_depth=max_depth, random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    print("Model training complete.")
    return clf

def evaluate_model(clf, X_test, y_test, class_names, reports_dir):
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=class_names, labels=[0, 1])
    reports_dir.mkdir(exist_ok=True, parents=True)
    report_path = reports_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(f"Saved classification report to {report_path}")

def plot_feature_importance(clf, X, plots_dir):
    print("Plotting feature importances...")
    plt.figure(figsize=(16, 8))
    plt.title('XGBoost Feature Importances (Top 20)', fontsize=20)
    plot_importance(clf, max_num_features=20, height=0.6, importance_type='gain', show_values=False)
    plt.tight_layout()
    plots_dir.mkdir(exist_ok=True, parents=True)
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
    clf = train_xgboost(X_train, y_train, max_depth)
    class_names = [f'not {mode}', f'{mode} (not preselected)']
    plots_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/xg_boost/mode_{mode}/max_depth_{max_depth}')
    reports_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/reports/xg_boost/mode_{mode}/max_depth_{max_depth}')
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
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
    main(mode_list=['standard_saver', 'fastpass', 'standard', 'premium', 'plus', 'lux', 'luxsuv'], max_depth_list=[10]) 