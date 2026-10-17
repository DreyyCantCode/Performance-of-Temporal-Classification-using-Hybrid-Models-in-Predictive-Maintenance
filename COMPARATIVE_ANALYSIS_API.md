# Comparative Analysis API Guide

This guide explains how to use the new comparative asset prediction API endpoints for analyzing and validating asset predictions.

## Overview

The system now includes several endpoints for comparative analysis of predictions from multiple models:

1. **Model Comparison Endpoints** - Compare predictions for specific assets or all assets
2. **Validation Endpoint** - Validate new asset data before making predictions
3. **Report Generation** - Generate comprehensive analysis reports

## API Endpoints

### 1. Compare Models for a Specific Asset

**Endpoint:** `GET /api/analysis/compare-models/<asset_id>`

**Purpose:** Compare predictions from the three hybrid variants (Hybrid V1, Hybrid V2, Hybrid V3) for a specific asset.

**Parameters:**
- `asset_id` (path parameter, required): The ID of the asset to analyze
- `horizon_hours` (query parameter, optional): Forecast horizon in hours (default: 24)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/analysis/compare-models/ASSET_01"
```

**Example Response:**
```json
{
  "asset_id": "ASSET_01",
  "timestamp": "2026-05-03T10:30:00.123456",
  "record_count": 45,
  "threshold": 0.5,
  "models": {
    "xgboost_only": {
      "probability": 0.65,
      "prediction": 1,
      "label": "Standalone XGBoost"
    },
    "bilstm_only": {
      "probability": 0.58,
      "prediction": 1,
      "label": "Standalone BiLSTM"
    },
    "hybrid": {
      "probability": 0.62,
      "prediction": 1,
      "label": "Stacked BiLSTM+XGBoost"
    }
  },
  "consensus": {
    "probability": 0.6166,
    "agreement_percentage": 100.0,
    "high_risk_models": ["xgboost_only", "bilstm_only", "hybrid"],
    "agreement_level": "High"
  },
  "trend": {
    "recent_change": 0.025,
    "direction": "increasing",
    "velocity": 0.025
  },
  "recommendation": "🔴 HIGH RISK: Multiple models predict high failure probability. Schedule maintenance soon."
}
```

**Status Codes:**
- `200 OK` - Analysis successful
- `404 Not Found` - Asset not found in data
- `400 Bad Request` - Data validation failed
- `500 Internal Server Error` - Server error

---

### 2. Compare All Assets

**Endpoint:** `GET /api/analysis/compare-all`

**Purpose:** Compare predictions across multiple assets.

**Parameters:**
- `limit` (query parameter, optional): Maximum number of assets to analyze (default: 50, max: 500)

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/analysis/compare-all?limit=20"
```

**Example Response:**
```json
{
  "total_assets": 20,
  "high_risk_assets": 3,
  "average_consensus_probability": 0.42,
  "average_agreement": 88.5,
  "predictions": [
    {
      "asset_id": "ASSET_01",
      "timestamp": "2026-05-03T10:30:00.123456",
      "record_count": 45,
      "consensus": {
        "probability": 0.62,
        "agreement_percentage": 100.0,
        "high_risk_models": ["xgboost_only", "bilstm_only"],
        "agreement_level": "High"
      }
    }
  ]
}
```

---

### 3. Generate Comparative Report

**Endpoint:** `GET /api/analysis/report`

**Purpose:** Generate a comprehensive report comparing all models across all assets.

**Parameters:** None

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/analysis/report"
```

**Example Response:**
```json
{
  "generated_at": "2026-05-03T10:35:00.123456",
  "total_records": 1250,
  "unique_assets": 25,
  "model_statistics": {
    "xgboost_only": {
      "mean_probability": 0.45,
      "std_probability": 0.22,
      "min_probability": 0.01,
      "max_probability": 0.98,
      "positive_rate": 0.38,
      "total_predictions": 1250
    },
    "bilstm_only": {
      "mean_probability": 0.42,
      "std_probability": 0.25,
      "min_probability": 0.02,
      "max_probability": 0.99,
      "positive_rate": 0.35,
      "total_predictions": 1250
    },
    "hybrid": {
      "mean_probability": 0.44,
      "std_probability": 0.23,
      "min_probability": 0.01,
      "max_probability": 0.97,
      "positive_rate": 0.37,
      "total_predictions": 1250
    }
  },
  "model_agreement": {
    "XGBoost Only vs BiLSTM Only": 85.6,
    "XGBoost Only vs Hybrid": 92.3,
    "BiLSTM Only vs Hybrid": 88.1
  },
  "risk_distribution": {
    "low_risk": 312,
    "moderate_risk": 625,
    "high_risk": 234,
    "critical_risk": 79
  },
  "threshold": 0.5
}
```

---

### 4. Validate New Asset Data

**Endpoint:** `POST /api/analysis/validate`

**Purpose:** Validate that new asset data can be used for prediction before adding it to the system.

**Request Body (JSON):**
```json
{
  "asset_id": "ASSET_NEW_01",
  "timestamp": "2026-05-03T10:00:00",
  "metrics": {
    "temperature_c": 45.5,
    "uptime_hours": 8.0,
    "workload_intensity": 0.75
  }
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/analysis/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "ASSET_NEW_01",
    "timestamp": "2026-05-03T10:00:00",
    "metrics": {
      "temperature_c": 45.5,
      "uptime_hours": 8.0,
      "workload_intensity": 0.75
    }
  }'
```

**Successful Response (200 OK):**
```json
{
  "valid": true,
  "asset_id": "ASSET_NEW_01",
  "timestamp": "2026-05-03T10:00:00",
  "message": "Data is valid and ready for prediction"
}
```

**Validation Error Response (400 Bad Request):**
```json
{
  "valid": false,
  "error": "Missing required metrics: ['workload_intensity']",
  "required_metrics": ["temperature_c", "uptime_hours", "workload_intensity"]
}
```

**Invalid Value Response:**
```json
{
  "valid": false,
  "errors": [
    "temperature_c value 500 out of reasonable range",
    "workload_intensity value is not numeric"
  ]
}
```

---

## Usage Examples

### Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# 1. Validate new asset data
validation_data = {
    "asset_id": "ASSET_NEW_01",
    "timestamp": "2026-05-03T10:00:00",
    "metrics": {
        "temperature_c": 45.5,
        "uptime_hours": 8.0,
        "workload_intensity": 0.75
    }
}

response = requests.post(f"{BASE_URL}/api/analysis/validate", json=validation_data)
print("Validation Result:", response.json())

# 2. Compare models for specific asset
if response.json()['valid']:
    asset_id = "ASSET_NEW_01"
    response = requests.get(f"{BASE_URL}/api/analysis/compare-models/{asset_id}")
    print("Model Comparison:", response.json())

# 3. Get all asset comparisons
response = requests.get(f"{BASE_URL}/api/analysis/compare-all?limit=10")
print("All Assets:", response.json())

# 4. Generate report
response = requests.get(f"{BASE_URL}/api/analysis/report")
report = response.json()
print(f"Total Assets: {report['total_assets']}")
print(f"High Risk Assets: {report['high_risk_assets']}")
print(f"Model Agreement: {report['model_agreement']}")
```

### JavaScript/Frontend Example

```javascript
// Async function to compare models
async function compareModels(assetId) {
  try {
    const response = await fetch(`/api/analysis/compare-models/${assetId}`);
    const data = await response.json();
    
    if (response.ok) {
      console.log('Comparison:', data);
      // Process and display data
    } else {
      console.error('Error:', data.error);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
}

// Validate new asset data
async function validateAssetData(assetData) {
  try {
    const response = await fetch('/api/analysis/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(assetData)
    });
    
    const result = await response.json();
    
    if (result.valid) {
      console.log('Data is valid and ready for prediction');
    } else {
      console.log('Validation errors:', result.errors);
    }
    
    return result.valid;
  } catch (error) {
    console.error('Validation failed:', error);
    return false;
  }
}
```

---

## Interpretation Guide

### Agreement Levels

- **High (≥80%)**: All models agree on the prediction. Decision is highly confident.
- **Moderate (50-80%)**: Most models agree. Decision is reasonably confident.
- **Low (<50%)**: Models disagree significantly. Additional investigation recommended.

### Risk Classification

- **Low Risk** (<0.3): Normal operation, routine monitoring sufficient
- **Moderate Risk** (0.3-0.6): Monitor closely, prepare maintenance schedule
- **High Risk** (0.6-0.8): Schedule maintenance soon, increase monitoring
- **Critical Risk** (≥0.8): Immediate maintenance required, high failure probability

### Trend Indicators

- **Increasing**: Risk is rising, trend toward failure
- **Decreasing**: Risk is improving, asset stabilizing
- **Stable**: Risk is steady, continue current monitoring

---

## Error Handling

### Common Errors

1. **Asset Not Found (404)**
   - Asset ID doesn't exist in current data
   - Solution: Check asset ID spelling, upload data with the asset

2. **Missing Required Fields (400)**
   - Validation endpoint requires all three metrics
   - Solution: Provide temperature_c, uptime_hours, workload_intensity

3. **No Data Loaded (500)**
   - System hasn't loaded any data yet
   - Solution: Upload a CSV file via `/api/data/upload`

4. **Invalid CSV Format (400)**
   - Uploaded file is missing required columns
   - Solution: Ensure CSV has 'asset_id' and 'timestamp' columns

---

## Best Practices

1. **Always validate new data before use**: Use `/api/analysis/validate` before adding new assets
2. **Monitor agreement levels**: Low agreement indicates need for model retraining
3. **Check trends regularly**: Track risk changes to predict maintenance needs
4. **Cross-reference models**: When models disagree, investigate the asset manually
5. **Generate reports periodically**: Monitor overall system health and model performance

---

## Integration with Existing Endpoints

### Upload New Data with Validation

```bash
# 1. Upload CSV file
curl -X POST "http://localhost:5000/api/data/upload" \
  -F "file=@new_data.csv"

# 2. Generate comparison report
curl -X GET "http://localhost:5000/api/analysis/report"

# 3. Compare specific assets
curl -X GET "http://localhost:5000/api/analysis/compare-all?limit=50"
```

### Register New Equipment and Validate

```bash
# 1. Register new equipment
curl -X POST "http://localhost:5000/api/equipment" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "NEW_ASSET_01",
    "name": "New Asset",
    "description": "Asset description",
    "room": "Lab 1",
    "shift_type": "am"
  }'

# 2. Validate data for this asset
curl -X POST "http://localhost:5000/api/analysis/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "NEW_ASSET_01",
    "timestamp": "2026-05-03T10:00:00",
    "metrics": {
      "temperature_c": 45.5,
      "uptime_hours": 8.0,
      "workload_intensity": 0.75
    }
  }'
```

---

## Troubleshooting

### Models Not Comparing
- **Cause**: One or more models not trained/loaded
- **Solution**: Check `/api/health` to see which models are loaded

### No Data Available
- **Cause**: No CSV data has been uploaded
- **Solution**: Upload a CSV file using `/api/data/upload` with proper format

### Predictions Unavailable
- **Cause**: Missing required feature columns
- **Solution**: Ensure data contains 'temperature_c', 'uptime_hours', 'workload_intensity'

### Validation Always Fails
- **Cause**: Incorrect metric names or data types
- **Solution**: Check metric names match exactly (case-sensitive)

For more help, check the application health endpoint: `GET /api/health`
