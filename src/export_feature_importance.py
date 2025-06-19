import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os

def export_feature_importance():
    """Export feature importance from XGBoost models to CSV files."""
    # Walk through the models directory
    for root, _, files in os.walk('models/xg_boost'):
        for file in files:
            if not file.endswith('_model.joblib'):
                continue
            
            model_path = Path(os.path.join(root, file))
            
            # Skip if not max_depth_10
            if 'max_depth_10' not in str(model_path):
                continue
            
            try:
                # Load the model
                model = joblib.load(model_path)
                
                # Get feature names and importance scores
                feature_names = model.get_booster().feature_names
                importance_scores = model.feature_importances_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_scores
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # Create corresponding output path in plots directory
                output_path = Path(str(model_path).replace('models', 'plots').replace('_model.joblib', '_feature_importance.csv'))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save to CSV
                importance_df.to_csv(output_path, index=False)
                print(f"Exported feature importance to {output_path}")
                
            except Exception as e:
                print(f"Error processing {model_path}: {e}")

if __name__ == "__main__":
    export_feature_importance() 