import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from utils.load_data import load_parquet_data
from pathlib import Path

# --- Helper: Churned user indicator (copied from churned_rider_analysis.py) ---
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
    df['target_diff_mode'] = (df['requested_ride_type'] != df['preselected_mode']).astype(int)
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
    print("Splitting data into train and test sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_xgboost(X_train, y_train, max_depth):
    print(f"Training XGBoost model (max_depth={max_depth})...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = XGBClassifier(max_depth=max_depth, random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
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

def plot_feature_importance(clf, X, plots_dir):
    print("Plotting feature importances...")
    plt.figure(figsize=(16, 8))
    plt.title('XGBoost Feature Importances (Top 20)', fontsize=20)
    plot_importance(clf, max_num_features=20, height=0.6, importance_type='gain', show_values=False)
    plt.tight_layout()
    plots_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(plots_dir / 'feature_importance.svg')
    print(f"Saved feature importance plot to {plots_dir / 'feature_importance.svg'}")

def main(max_depth=5):
    X, y, feature_cols = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_xgboost(X_train, y_train, max_depth)
    class_names = ['requested preselected mode', 'requested non-preselected mode']
    plots_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/plots/xg_boost/max_depth_{max_depth}')
    reports_dir = Path(f'/home/sagemaker-user/studio/src/new-rider-v3/reports/xg_boost/max_depth_{max_depth}')
    evaluate_model(clf, X_test, y_test, class_names, reports_dir)
    plot_feature_importance(clf, X, plots_dir)

if __name__ == "__main__":
    main(6) 