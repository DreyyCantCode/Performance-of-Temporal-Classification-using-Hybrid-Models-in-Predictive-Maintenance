# Metric Classification System

## Overview
The system accurately classifies asset health status using **4 severity levels** based on multiple metrics and ML predictions.

## Classification Levels

### 1. 🟢 **Normal** (Priority: 1)
- **Health Score**: ≥ 85%
- **Performance Score**: ≥ 85%
- **Temperature**: ≤ 45°C
- **Failure Risk**: < 50%
- **Color**: Primary (Green)
- **Action**: Continue monitoring

### 2. 🟡 **Warning** (Priority: 2)
- **Health Score**: 60-85%
- **Performance Score**: 60-85%
- **Temperature**: 45-75°C
- **Failure Risk**: 50-70%
- **Color**: Warning (Yellow)
- **Action**: Schedule preventive maintenance

### 3. 🔴 **Danger** (Priority: 3)
- **Health Score**: 40-60%
- **Performance Score**: 40-60%
- **Temperature**: 75-90°C
- **Failure Risk**: 70-90%
- **Color**: Danger (Red)
- **Action**: Immediate maintenance required

### 4. ⚠️ **Critical** (Priority: 4)
- **Health Score**: < 40%
- **Performance Score**: < 40%
- **Temperature**: > 90°C
- **Failure Risk**: > 90%
- **Color**: Yellow (Critical Alert)
- **Action**: Emergency intervention - Risk of imminent failure

## Classification Logic

The system uses a **worst-case approach**:
- Status is determined by the **most severe** metric
- If ANY metric is Critical → Asset is Critical
- If ANY metric is Danger (and none Critical) → Asset is Danger
- If ANY metric is Warning (and none worse) → Asset is Warning
- Only if ALL metrics are good → Asset is Normal

## Configuration

All thresholds are configurable in `config/metrics.json`:

```json
{
  "metrics": [
    {
      "name": "health_score",
      "critical_threshold": 40,
      "bad_threshold": 60,
      "warning_threshold": 75,
      "good_threshold": 85
    },
    {
      "name": "performance_score",
      "critical_threshold": 40,
      "bad_threshold": 60,
      "warning_threshold": 75,
      "good_threshold": 85
    },
    {
      "name": "temperature_c",
      "good_threshold": 45,
      "warning_threshold": 60,
      "bad_threshold": 75,
      "critical_threshold": 90
    }
  ],
  "prediction_model": {
    "high_risk_threshold": 0.7,
    "critical_risk_threshold": 0.9
  }
}
```

## API Usage

### Get Classifications with Predictions
```bash
GET /api/predictions/latest
```

Response includes classification for each asset:
```json
{
  "predictions": [
    {
      "asset_id": "PC-001",
      "health_score": 65.5,
      "performance_score": 72.0,
      "temperature_c": 58.3,
      "failure_risk": 0.45,
      "status": "Warning",
      "status_color": "warning",
      "status_priority": 2,
      "status_issues": [
        "Health below optimal (65.5%)",
        "Temperature elevated (58.3°C)"
      ],
      "health_status": "Warning",
      "performance_status": "Warning",
      "temperature_status": "Elevated",
      "risk_status": "Low"
    }
  ]
}
```

### Classify Custom Metrics
```bash
POST /api/classify
Content-Type: application/json

{
  "health_score": 75.5,
  "performance_score": 82.0,
  "temperature_c": 55.3,
  "failure_risk": 0.45
}
```

Response:
```json
{
  "metrics": {
    "health_score": 75.5,
    "performance_score": 82.0,
    "temperature_c": 55.3,
    "failure_risk": 0.45
  },
  "classification": {
    "status": "Warning",
    "color": "warning",
    "priority": 2,
    "issues": [
      "Health below optimal (75.5%)",
      "Temperature elevated (55.3°C)"
    ],
    "health_status": "Warning",
    "performance_status": "Good",
    "temperature_status": "Elevated",
    "risk_status": "Low"
  },
  "thresholds": {
    "health": {
      "critical": 40,
      "bad": 60,
      "warning": 75,
      "good": 85
    },
    ...
  }
}
```

## Maintenance Impact

When maintenance is completed:
1. New health metrics are recorded
2. System creates new sensor reading with updated values
3. Next predictions use the improved metrics
4. Asset status automatically updates based on new metrics

Example flow:
```
Before Maintenance:
- Health: 45% → Status: Danger
- Perform maintenance, clean filters, replace parts
After Maintenance:
- Health: 88% → Status: Normal
- New predictions reflect improved health
```

## Best Practices

1. **Regular Monitoring**: Check dashboard for Warning/Danger/Critical assets
2. **Preventive Action**: Address Warning status before it becomes Danger
3. **Priority Response**: Always handle Critical status immediately
4. **Record Keeping**: Complete maintenance records with before/after metrics
5. **Threshold Tuning**: Adjust thresholds based on your equipment specifications

## Threshold Customization Guide

### For Lab Equipment (PCs, Projectors):
- **Health**: Critical < 40, Warning < 75, Good ≥ 85
- **Temperature**: Critical > 80°C, Warning > 50°C, Good ≤ 40°C

### For Industrial Equipment:
- **Health**: Critical < 50, Warning < 70, Good ≥ 85
- **Temperature**: Critical > 100°C, Warning > 70°C, Good ≤ 60°C

### For Precision Instruments:
- **Health**: Critical < 60, Warning < 80, Good ≥ 90
- **Temperature**: Critical > 40°C, Warning > 30°C, Good ≤ 25°C

Modify `config/metrics.json` to match your specific requirements!
