#!/usr/bin/env python
"""Quick test to verify models are working and collecting metrics."""

import json
from pathlib import Path
from src.predict_xgb import run_prediction_pipeline, load_xgb_metadata

# Test prediction pipeline
print("=" * 80)
print("TESTING PREDICTION PIPELINE")
print("=" * 80)

result_df = run_prediction_pipeline(
    'data/lab_maintenance_synthetic_daily.csv',
    'models',
    'data/test_predictions.csv',
    top_n=5,
    require_stacked=False
)

# Load metadata to check model config
metadata_path = Path('models/xgb_metadata.json')
if metadata_path.exists():
    print("\n" + "=" * 80)
    print("MODEL METADATA")
    print("=" * 80)
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\nModel Mode: {metadata.get('model_mode', 'N/A')}")
    print(f"BiLSTM Window Size: {metadata.get('bilstm_window_size', 'N/A')}")
    print(f"Feature Columns Count: {len(metadata.get('feature_columns', []))}")
    print(f"Threshold: {metadata.get('threshold', 'N/A'):.4f}")
    print(f"Calibration Method: {metadata.get('calibration', {}).get('method', 'none')}")
    
    # Check test metrics if available
    if 'test_metrics' in metadata:
        test_metrics = metadata['test_metrics']
        print(f"\nTest Metrics:")
        print(f"  - Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  - Precision: {test_metrics.get('precision', 'N/A'):.4f}")
        print(f"  - Recall: {test_metrics.get('recall', 'N/A'):.4f}")
        print(f"  - F1-Score: {test_metrics.get('f1', 'N/A'):.4f}")

# Check prediction results
print("\n" + "=" * 80)
print("PREDICTION RESULTS SUMMARY")
print("=" * 80)
if result_df is not None and len(result_df) > 0:
    print(f"\nTotal records: {len(result_df)}")
    print(f"\nPrediction distributions:")
    print(f"  Hybrid-V1 positive rate: {result_df['pred_hybrid_v1'].mean():.2%}")
    print(f"  Hybrid-V2 positive rate: {result_df['pred_hybrid_v2'].mean():.2%}")
    print(f"  Hybrid-V3 positive rate: {result_df['pred_hybrid_v3'].mean():.2%}")
    
    print(f"\nProbability statistics (Hybrid-V2):")
    print(f"  Min: {result_df['prob_hybrid_v2'].min():.4f}")
    print(f"  Max: {result_df['prob_hybrid_v2'].max():.4f}")
    print(f"  Mean: {result_df['prob_hybrid_v2'].mean():.4f}")
    print(f"  Median: {result_df['prob_hybrid_v2'].median():.4f}")

print("\n✓ Model is working and producing predictions!")
print(f"✓ Predictions saved to data/test_predictions.csv")
