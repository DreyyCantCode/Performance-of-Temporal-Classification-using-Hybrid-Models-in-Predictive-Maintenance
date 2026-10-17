# Quick Start: New Comparative Analysis Features

This guide helps you quickly get started with the new asset prediction comparative analysis features.

## What's New

✅ **Comparative Analysis** - Compare predictions from three hybrid BiLSTM+XGBoost variants  
✅ **Data Validation** - Validate new asset data before predictions  
✅ **Asset Registration** - Improved equipment registration with validation  
✅ **Comprehensive Reports** - Generate system-wide analysis reports  

## 5-Minute Setup

### 1. Start the Dashboard

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start the Flask dashboard
python -m app.py
# or
.\start_dashboard.ps1
```

Dashboard available at: `http://localhost:5000`

### 2. Test the New Endpoints

#### Quick Test with PowerShell

```powershell
# Get health status
Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method GET

# Generate comparison report
Invoke-RestMethod -Uri "http://localhost:5000/api/analysis/report" -Method GET

# Compare all assets
Invoke-RestMethod -Uri "http://localhost:5000/api/analysis/compare-all?limit=10" -Method GET
```

#### Or with curl (if installed)

```bash
# Get health status
curl http://localhost:5000/api/health

# Generate comparison report
curl http://localhost:5000/api/analysis/report

# Compare specific asset
curl http://localhost:5000/api/analysis/compare-models/ASSET_01
```

### 3. Validate New Asset Data

```powershell
$data = @{
    asset_id = "NEW_ASSET_01"
    timestamp = "2026-05-03T10:00:00"
    metrics = @{
        temperature_c = 45.5
        uptime_hours = 8.0
        workload_intensity = 0.75
    }
}

$json = $data | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/analysis/validate" `
    -Method POST `
    -ContentType "application/json" `
    -Body $json
```

### 4. Register New Equipment

```powershell
$equipment = @{
    asset_id = "NEW_ASSET_01"
    name = "New Equipment"
    description = "Equipment for testing"
    room = "Lab 1"
    shift_type = "am"
    status = "active"
}

$json = $equipment | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/equipment" `
    -Method POST `
    -ContentType "application/json" `
    -Body $json
```

## Common Workflows

### Workflow 1: Analyze a Single Asset

```bash
# Get detailed comparison for one asset
curl "http://localhost:5000/api/analysis/compare-models/ASSET_01"

# Expected response includes:
# - Predictions from all 3 hybrid variants
# - Consensus probability and agreement level
# - Risk trend and direction
# - Maintenance recommendation
```

### Workflow 2: Upload New Data and Compare

```bash
# 1. Upload CSV file
curl -X POST "http://localhost:5000/api/data/upload" \
  -F "file=@new_maintenance_data.csv"

# 2. Generate report
curl "http://localhost:5000/api/analysis/report"

# 3. Check high-risk assets
curl "http://localhost:5000/api/analysis/compare-all?limit=50"
```

### Workflow 3: Register Asset and Validate Data

```bash
# 1. Register equipment
curl -X POST "http://localhost:5000/api/equipment" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "NEW_01",
    "name": "New Device",
    "description": "Device description",
    "room": "Lab 1",
    "shift_type": "am"
  }'

# 2. Validate prediction data
curl -X POST "http://localhost:5000/api/analysis/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "NEW_01",
    "timestamp": "2026-05-03T10:00:00",
    "metrics": {
      "temperature_c": 45.5,
      "uptime_hours": 8.0,
      "workload_intensity": 0.75
    }
  }'

# 3. Get comparison for the new asset
curl "http://localhost:5000/api/analysis/compare-models/NEW_01"
```

## Understanding the Results

### Model Predictions

Each model returns a failure probability (0-1):

```
Hybrid V1: 0.65 (65% failure probability)
Hybrid V2: 0.58 (58% failure probability)
Hybrid V3: 0.62 (62% failure probability)
```

### Consensus Analysis

- **Probability**: Average of all model predictions
- **Agreement %**: How many models agree on failure/success
- **Agreement Level**: 
  - **High** (≥80%): Very confident
  - **Moderate** (50-80%): Reasonably confident
  - **Low** (<50%): Uncertain, investigate further

### Recommendations

| Probability | Agreement | Recommendation |
|------------|-----------|----------------|
| >0.8 | >66% | 🔴 CRITICAL: Immediate maintenance needed |
| >0.7 | Multiple high-risk | 🟠 HIGH RISK: Schedule maintenance soon |
| >0.5 | >50% | 🟡 ELEVATED: Monitor closely |
| <0.5 | Low risk | 🟢 NORMAL: Continue routine monitoring |

## Checking System Health

```bash
# Get complete system status
curl http://localhost:5000/api/health

# Returns information about:
# - Model status (XGBoost, BiLSTM loaded?)
# - Data loaded
# - Thresholds and parameters
# - Performance metrics
```

## Troubleshooting

### Issue: "No data loaded"

```bash
# Solution: Upload data first
curl -X POST "http://localhost:5000/api/data/upload" \
  -F "file=@data/lab_maintenance_full.csv"
```

### Issue: Asset not found

```bash
# Check available assets
curl "http://localhost:5000/api/config/assets"

# Or check data
curl "http://localhost:5000/api/data/logs" | grep "ASSET_ID"
```

### Issue: Validation fails

```bash
# Check required fields and metrics
curl -X POST "http://localhost:5000/api/analysis/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "ASSET_01",
    "timestamp": "2026-05-03T10:00:00",
    "metrics": {
      "temperature_c": 45.5,
      "uptime_hours": 8.0,
      "workload_intensity": 0.75
    }
  }'
```

### Issue: Models not comparing

```bash
# Check health status
curl http://localhost:5000/api/health | grep -i "loaded"

# If models not loaded, check logs
# Models should be in models/ directory:
# - models/xgb_model.json
# - models/bilstm.h5
# - models/xgb_metadata.json
```

## Next Steps

1. **Read Full Documentation**: See [COMPARATIVE_ANALYSIS_API.md](./COMPARATIVE_ANALYSIS_API.md)
2. **Run Tests**: `python test_backend.py`
3. **Train Models**: `.\train_model.ps1`
4. **Monitor Assets**: Use dashboard at http://localhost:5000
5. **Explore Predictions**: Call comparative analysis endpoints

## Key Files

- **Backend**: `app.py` - Main Flask application
- **Analysis Module**: `src/comparative_analysis.py` - Comparison logic
- **Documentation**: `COMPARATIVE_ANALYSIS_API.md` - Full API reference
- **Tests**: `test_backend.py` - Validation tests

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analysis/compare-models/<asset_id>` | GET | Compare models for one asset |
| `/api/analysis/compare-all` | GET | Compare across all assets |
| `/api/analysis/report` | GET | Generate comprehensive report |
| `/api/analysis/validate` | POST | Validate new asset data |
| `/api/equipment` | POST | Register new equipment |
| `/api/data/upload` | POST | Upload new CSV data |

## Questions?

- Check the [full API documentation](./COMPARATIVE_ANALYSIS_API.md)
- Run `python test_backend.py` to verify everything works
- Check application logs: `app.logger.info()` in dashboard terminal
- Review model health: `GET /api/health`

## Success Indicators ✅

You'll know everything is working when:

1. ✅ Dashboard loads at `http://localhost:5000`
2. ✅ `/api/health` returns all models loaded
3. ✅ `/api/analysis/report` returns statistics
4. ✅ `/api/analysis/validate` validates data correctly
5. ✅ `/api/analysis/compare-models/<asset>` shows predictions
6. ✅ `test_backend.py` passes all tests

---

**Ready to use comparative asset predictions!** 🚀
