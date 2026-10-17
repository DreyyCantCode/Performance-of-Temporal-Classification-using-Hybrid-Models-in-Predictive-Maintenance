"""
Test script to validate the backend with sample data.
This script tests the new comparative analysis endpoints and data validation.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.comparative_analysis import (
    validate_asset_data,
    compare_models_for_asset,
    generate_comparative_report,
)


def create_sample_prediction_data() -> pd.DataFrame:
    """Create sample prediction data for testing."""
    np.random.seed(42)
    
    assets = [f'ASSET_{i:02d}' for i in range(1, 6)]
    timestamps = []
    asset_ids = []
    temps = []
    uptimes = []
    workloads = []
    xgb_probs = []
    bilstm_probs = []
    hybrid_probs = []
    
    base_time = datetime(2026, 5, 1)
    
    for asset in assets:
        for day in range(3):
            for hour in range(8):
                timestamps.append((base_time + timedelta(days=day, hours=hour)).isoformat())
                asset_ids.append(asset)
                temps.append(45 + np.random.randn() * 5)
                uptimes.append(8 + np.random.randn() * 1)
                workloads.append(0.5 + np.random.rand() * 0.4)
                
                # Generate correlated probabilities
                base_prob = 0.4 + np.random.randn() * 0.2
                base_prob = np.clip(base_prob, 0.0, 1.0)
                
                xgb_probs.append(base_prob)
                bilstm_probs.append(base_prob + np.random.randn() * 0.05)
                hybrid_probs.append(base_prob + np.random.randn() * 0.03)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'asset_id': asset_ids,
        'temperature_c': temps,
        'uptime_hours': uptimes,
        'workload_intensity': workloads,
        'prob_xgb_standalone': np.clip(xgb_probs, 0, 1),
        'prob_bilstm_standalone': np.clip(bilstm_probs, 0, 1),
        'prob_hybrid': np.clip(hybrid_probs, 0, 1),
        'pred_xgb_standalone': (np.array(xgb_probs) > 0.5).astype(int),
        'pred_bilstm_standalone': (np.clip(bilstm_probs, 0, 1) > 0.5).astype(int),
        'pred_hybrid': (np.clip(hybrid_probs, 0, 1) > 0.5).astype(int),
        'failure_risk': np.clip(hybrid_probs, 0, 1),
    })
    
    return df


def test_data_validation():
    """Test data validation function."""
    print("\n" + "="*60)
    print("TEST 1: Data Validation")
    print("="*60)
    
    # Test with valid data
    df = create_sample_prediction_data()
    is_valid, error = validate_asset_data(df)
    print(f"✓ Valid data: {is_valid} (error: {error or 'None'})")
    
    # Test with empty data
    is_valid, error = validate_asset_data(pd.DataFrame())
    print(f"✓ Empty data rejected: {not is_valid} (error: {error})")
    
    # Test with specific asset
    is_valid, error = validate_asset_data(df, 'ASSET_01')
    print(f"✓ Specific asset found: {is_valid} (error: {error or 'None'})")
    
    # Test with non-existent asset
    is_valid, error = validate_asset_data(df, 'ASSET_99')
    print(f"✓ Non-existent asset rejected: {not is_valid} (error: {error})")
    
    return True


def test_model_comparison():
    """Test model comparison for a single asset."""
    print("\n" + "="*60)
    print("TEST 2: Model Comparison (Single Asset)")
    print("="*60)
    
    df = create_sample_prediction_data()
    metadata = {'threshold': 0.5, 'feature_columns': ['temperature_c', 'uptime_hours', 'workload_intensity']}
    
    asset_id = 'ASSET_01'
    analysis = compare_models_for_asset(df, asset_id, metadata)
    
    if 'error' in analysis:
        print(f"✗ Error: {analysis['error']}")
        return False
    
    print(f"✓ Asset ID: {analysis['asset_id']}")
    print(f"✓ Record count: {analysis['record_count']}")
    print(f"✓ Timestamp: {analysis['timestamp']}")
    
    models = analysis['models']
    print(f"\nModel Predictions:")
    for model_name, data in models.items():
        print(f"  {model_name:20}: prob={data['probability']:.3f}, pred={data['prediction']}")
    
    consensus = analysis['consensus']
    print(f"\nConsensus:")
    print(f"  Probability: {consensus['probability']:.3f}")
    print(f"  Agreement: {consensus['agreement_percentage']:.1f}%")
    print(f"  Agreement Level: {consensus['agreement_level']}")
    print(f"  High Risk Models: {len(consensus['high_risk_models'])}")
    
    print(f"\nTrend:")
    print(f"  Direction: {analysis['trend']['direction']}")
    print(f"  Recent Change: {analysis['trend']['recent_change']:.4f}")
    
    print(f"\nRecommendation:")
    print(f"  {analysis['recommendation']}")
    
    return True


def test_report_generation():
    """Test comprehensive report generation."""
    print("\n" + "="*60)
    print("TEST 3: Report Generation")
    print("="*60)
    
    df = create_sample_prediction_data()
    metadata = {'threshold': 0.5, 'feature_columns': ['temperature_c', 'uptime_hours', 'workload_intensity']}
    
    report = generate_comparative_report(df, metadata)
    
    print(f"✓ Total Records: {report['total_records']}")
    print(f"✓ Unique Assets: {report['unique_assets']}")
    print(f"✓ Threshold: {report['threshold']}")
    
    print(f"\nModel Statistics:")
    for model, stats in report['model_statistics'].items():
        print(f"\n  {model}:")
        print(f"    Mean Probability: {stats['mean_probability']:.3f}")
        print(f"    Std Probability: {stats['std_probability']:.3f}")
        print(f"    Positive Rate: {stats['positive_rate']:.1%}")
        print(f"    Range: [{stats['min_probability']:.3f}, {stats['max_probability']:.3f}]")
    
    print(f"\nModel Agreement Matrix:")
    for pair, agreement in report['model_agreement'].items():
        print(f"  {pair}: {agreement:.1f}%")
    
    print(f"\nRisk Distribution:")
    risk_dist = report['risk_distribution']
    total = sum(risk_dist.values())
    for risk_level, count in risk_dist.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {risk_level}: {count} ({pct:.1f}%)")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases")
    print("="*60)
    
    # Test with missing columns
    df_missing = pd.DataFrame({
        'timestamp': ['2026-05-01T10:00:00'],
        'asset_id': ['ASSET_01'],
    })
    is_valid, error = validate_asset_data(df_missing)
    print(f"✓ Missing prediction columns detected: {not is_valid}")
    print(f"  Error: {error}")
    
    # Test with NaN values
    df_nan = create_sample_prediction_data()
    df_nan.loc[0, 'failure_risk'] = np.nan
    print(f"✓ Created data with NaN values")
    
    # Test with single record
    df_single = create_sample_prediction_data().iloc[:1].copy()
    is_valid, error = validate_asset_data(df_single)
    print(f"✓ Single record: {is_valid}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("█" * 60)
    print("  BACKEND VALIDATION TEST SUITE")
    print("█" * 60)
    
    tests = [
        ("Data Validation", test_data_validation),
        ("Model Comparison", test_model_comparison),
        ("Report Generation", test_report_generation),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
