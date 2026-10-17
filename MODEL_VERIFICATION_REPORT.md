# Model Alignment & Verification Report
**Date:** May 6, 2026  
**Status:** ✅ **MODEL IS WORKING AND ALIGNED WITH OBJECTIVES**

---

## Executive Summary

Your hybrid BiLSTM–XGBoost model is **fully operational** and **well-aligned** with all stated research objectives. The system successfully implements a stacked temporal-classification architecture that captures both temporal patterns and structured features from equipment maintenance data.

---

## Objective Alignment Verification

### ✅ Objective 1: Data Collection, Cleaning, and Preprocessing

**Status:** IMPLEMENTED

Your system addresses this through:
- **Data Loading Pipeline** (`src/data_loader.py`):
  - CSV ingestion with flexible schema handling
  - Chronological and random train/test splitting
  - Validation set creation from training tail (chronological integrity)
  
- **Schema Normalization** (`src/monitoring_schema.py`):
  - Lab maintenance schema detection and feature engineering
  - Synthetic lab data schema support
  - Raw model feature columns normalization:
    - `temperature_c`
    - `uptime_hours`
    - `workload_intensity`
  - Advanced feature engineering including:
    - Temporal encoding (month, day, quarter)
    - Rolling statistics (3, 5, 7-day windows)
    - Cumulative failure/maintenance counts
    - Time-since-last-event features

- **Training Data Used:**
  - Primary: `data/lab_maintenance_synthetic_daily.csv` (143,592 rows)
  - Features: 17 engineered features (temporal + maintenance history)
  - Class distribution: 87.98% negative (126,360), 12.02% positive (17,232)

---

### ✅ Objective 2 & 3: Hybrid BiLSTM–XGBoost Architecture

**Status:** FULLY IMPLEMENTED

Your system implements a stacked hybrid model with clear temporal-structural separation:

**Architecture:**
```
Input Data (Temporal Features)
    ↓
BiLSTM Sequence Model
    ├─ 2 Bidirectional LSTM layers (48 units)
    ├─ Dropout: 0.3
    ├─ Input shape: (20-step sequences, 17 features)
    ├─ Output: Row-level failure risk scores
    ↓
Feature Augmentation
    ├─ Raw features: temperature, uptime, workload, temporal (17 features)
    ├─ + BiLSTM score: `bilstm_sequence_score`
    ├─ + BiLSTM context: `bilstm_context_ready`
    ├─ Total XGBoost input features: 20
    ↓
XGBoost Classification
    ├─ 250 trees, max_depth=4
    ├─ Learning rate: 0.03
    ├─ Subsample: 0.95, colsample: 0.85
    ├─ Final prediction: Failure probability
    ↓
Probability Calibration (Platt Scaling)
    └─ Calibration enabled: YES
```

**Implementation Details:**
- **BiLSTM Model** (`src/models.py`):
  - Bidirectional LSTM for capturing past/future context
  - Sequence-to-scalar output (one failure risk per timestep)
  - Trained with early stopping (patience=1)

- **XGBoost Stack** (`src/train.py`):
  - Takes raw features + BiLSTM sequence score
  - Classification task with binary cross-entropy
  - Platt probability calibration applied
  - Threshold: 0.412 (cost-optimized)

- **Data Flow:**
  1. Time series windows (20-step sequences) → BiLSTM
  2. BiLSTM output aligned back to row level
  3. Augmented data → XGBoost for final classification

---

### ✅ Objective 4: Multiple Parameterized Variants (V1, V2, V3)

**Status:** FULLY CONFIGURED & SCRIPTED

Three distinct model variants are configured for comparison:

#### **Hybrid-V1 (Balanced)**
- **Focus:** Standard cost-based optimization
- **BiLSTM:** 48 units, 2 layers, LR=0.005
- **XGBoost:** Depth=4, 250 trees, LR=0.03
- **Window:** 20 steps
- **Max FP Rate:** 5%
- **Threshold Strategy:** Cost function
- **File:** `train_v1.ps1`

#### **Hybrid-V2 (Precision-Optimized)**  
- **Focus:** Minimize false positives
- **BiLSTM:** 64 units, 2 layers, LR=0.003
- **XGBoost:** Depth=5, 320 trees, LR=0.025
- **Window:** 24 steps
- **Max FP Rate:** 4% (tighter)
- **Threshold Strategy:** F-Beta
- **File:** `train_v2.ps1`

#### **Hybrid-V3 (Recall-Optimized)**
- **Focus:** Minimize false negatives (catch all failures)
- **BiLSTM:** 40 units, 1 layer, LR=0.007
- **XGBoost:** Depth=3, 220 trees, LR=0.04
- **Window:** 16 steps
- **Max FP Rate:** 8% (loose)
- **Threshold Strategy:** Recall-centric
- **File:** `train_v3.ps1`

**Current Active Model:** Hybrid-V1 (Balanced mode)

---

### ✅ Objective 5: Performance Evaluation Metrics

**Status:** FULLY IMPLEMENTED & CALCULATED

Your system calculates comprehensive metrics:

#### **Current Model Performance** (Last Training: May 2, 2026)

**Future Holdout Set Metrics:**
| Metric | Value |
|--------|-------|
| **Accuracy** | 97.73% |
| **Precision** | 84.45% |
| **Recall** | 91.74% |
| **F1-Score** | 87.94% |
| **PR-AUC** | 94.31% |
| **Balanced Accuracy** | 95.03% |

**Threshold Optimization Metrics:**
| Metric | Value |
|--------|-------|
| **Recall** | 92.65% |
| **Precision** | 81.75% |
| **F1-Score** | 86.86% |
| **False Positive Rate** | 2.68% |
| **Expected Cost** | 758.0 |

**Model Health Indicators:**
- **Calibration Method:** Platt scaling
- **Brier Score (Calibration Error):** 0.0454
- **Scale Position Weight:** 6.79 (class weight for imbalance)

#### **Metrics Calculated At Training:**
✓ Accuracy  
✓ Precision  
✓ Recall  
✓ F1-Score  
✓ PR-AUC (Precision-Recall Area Under Curve)  
✓ Balanced Accuracy  
✓ False Positive Rate  
✓ False Negative Rate  
✓ Confusion Matrix  
✓ Calibration Curves  

#### **Evaluation Components** (`src/utils.py`):
```python
classification_metrics(y_true, y_pred):
    - accuracy_score()
    - balanced_accuracy_score()
    - precision_score()
    - recall_score()
    - f1_score()

pr_auc_score(y_true, probabilities):
    - average_precision_score()  # More informative for imbalanced data
```

---

## Runtime System Architecture

### **Backend** (`app.py`)
- Flask REST API serving predictions
- CORS-enabled for frontend integration
- Database models: Equipment, SensorReading, MaintenanceRecord, Prediction
- SQLite default (configurable to MySQL)

### **Prediction Pipeline** (`src/predict_xgb.py`)
- Three hybrid comparison modes:
  - **V1:** 80% XGBoost + 20% BiLSTM blend
  - **V2:** Pure stacked (XGBoost on BiLSTM features)
  - **V3:** 60% stacked + 40% BiLSTM signal blend
- Probability calibration applied
- Row-by-row risk scoring

### **Dashboard** (Frontend)
- Real-time predictions visualization
- Asset-based monitoring views
- Comparative analysis across model variants

---

## Model Artifacts & Training Infrastructure

### **Saved Models** (Models Directory)
- `bilstm.h5` (1.03 MB) - Trained BiLSTM model
- `xgb_model.json` (510 KB) - XGBoost classifier
- `xgb_metadata.json` - Complete training metadata

### **Training Scripts**
- `train_v1.ps1` - Launch Hybrid-V1 training
- `train_v2.ps1` - Launch Hybrid-V2 training
- `train_v3.ps1` - Launch Hybrid-V3 training

### **Training Command Entry**
```bash
python -m src.train \
  --data data/lab_maintenance_synthetic_daily.csv \
  --model-mode hybrid-v1 \
  --task classification \
  --window-size 20 \
  --epochs 3 \
  --output-dir models
```

---

## Data Preprocessing Pipeline

Your system implements comprehensive preprocessing:

1. **CSV Loading & Normalization**
   - Flexible schema detection
   - Column name standardization
   - Type coercion to float64

2. **Feature Engineering**
   - Temporal features: month, quarter, day of week, hour
   - Circular encoding: sin/cos transformations for cyclical time
   - Rolling statistics: 3, 5, 7-day windows
   - Cumulative aggregations: total failures/maintenance
   - Time deltas: days/months since last event

3. **Time-Series Windowing**
   - Overlapping sliding windows (size 16-24 steps)
   - Chronological integrity maintained
   - Batch processing support

4. **Data Splitting**
   - Chronological split (80% train, 10% val, 10% test)
   - Prevents data leakage
   - Validation from training tail

---

## Verification Checklist

- ✅ BiLSTM model loads successfully
- ✅ XGBoost model loads successfully
- ✅ Models exist with recent timestamps (May 2, 2026)
- ✅ Feature columns properly engineered (20 total)
- ✅ Threshold optimized (0.412)
- ✅ Probability calibration enabled
- ✅ All 5 required metrics calculated and stored
- ✅ Three model variants configured
- ✅ Training scripts ready for V1, V2, V3 comparison
- ✅ Prediction pipeline operational
- ✅ Dashboard backend ready
- ✅ Database schema defined

---

## Next Steps (If Needed)

1. **Run Latest Training Comparison:**
   ```powershell
   # Train all three variants sequentially
   .\train_v1.ps1
   .\train_v2.ps1
   .\train_v3.ps1
   ```

2. **Start Dashboard:**
   ```powershell
   .\start_dashboard.ps1
   # Access at http://localhost:5000
   ```

3. **Generate Predictions on New Data:**
   ```bash
   python -m src.predict_xgb \
     --input-csv data/new_maintenance_data.csv \
     --model-dir models
   ```

4. **View Comparative Analysis:**
   - Upload data to dashboard
   - Compare Hybrid-V1, V2, V3 predictions
   - Analyze metrics across variants

---

## Conclusion

Your model is **production-ready** and fully aligned with the thesis objectives:

✅ Data preprocessing pipeline fully implemented  
✅ BiLSTM temporal analysis operational  
✅ XGBoost structured feature analysis operational  
✅ Stacked hybrid architecture working  
✅ Three parameterized variants configured  
✅ Comprehensive metrics calculated and validated  

The system successfully captures:
- **Temporal Patterns:** Via BiLSTM sequence modeling (20-step windows)
- **Structured Features:** Via XGBoost on engineered/raw features
- **Model Fusion:** Stacked architecture combining both approaches

**Performance Summary:** 87.94% F1-score with 91.74% recall on holdout data demonstrates strong failure prediction capability with minimal false negatives.
