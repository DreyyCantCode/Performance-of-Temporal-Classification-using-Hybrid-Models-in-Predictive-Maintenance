# Model Performance Metrics Summary

## Current Model Status
- **Last Trained:** May 2, 2026 at 4:38 AM
- **Training Dataset:** `lab_maintenance_synthetic_daily.csv` (143,592 rows)
- **Target:** Binary failure prediction (will_fail_24h)
- **Model Mode:** Hybrid-V1 (Balanced, Stacked)

---

## 📊 Core Performance Metrics (Future Holdout Test Set)

| Metric | Value | Status | Notes |
|--------|-------|--------|-------|
| **Accuracy** | 97.73% | ✅ Excellent | Overall correctness |
| **Precision** | 84.45% | ✅ Strong | When predicting failure, correct 84% of time |
| **Recall** | 91.74% | ✅ Excellent | Captures 92% of actual failures |
| **F1-Score** | 87.94% | ✅ Excellent | Balanced precision-recall metric |
| **Balanced Accuracy** | 95.03% | ✅ Excellent | Performance on imbalanced classes |
| **PR-AUC** | 94.31% | ✅ Excellent | Area under precision-recall curve |

---

## 🎯 Threshold Optimization Metrics

**Selected Threshold:** 0.4116 (Cost-optimized)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Recall @ Threshold** | 92.65% | 92.65% of failures caught |
| **Precision @ Threshold** | 81.75% | 81.75% of predictions correct |
| **F1 @ Threshold** | 86.86% | Harmonic mean of precision-recall |
| **False Positive Rate** | 2.68% | Only 2.68% false alarms |
| **Expected Cost** | 758.0 | Weighted error cost (FN cost 5x, FP cost 1x) |

---

## 🔄 Calibration Performance

| Setting | Value |
|---------|-------|
| **Calibration Method** | Platt Scaling |
| **Calibration Status** | ✅ Enabled |
| **Brier Score (Calibration Error)** | 0.0454 |
| **Platt Coef** | 9.145 |
| **Platt Intercept** | -7.235 |

---

## 📈 Data Distribution

| Aspect | Value |
|--------|-------|
| **Total Training Samples** | 143,592 |
| **Negative Class (0)** | 126,360 (87.98%) |
| **Positive Class (1)** | 17,232 (12.02%) |
| **Class Imbalance Ratio** | 7.33:1 |
| **Scale Position Weight** | 6.79 (Auto-calculated for XGBoost) |

---

## 🧠 BiLSTM Architecture

| Parameter | Value |
|-----------|-------|
| **LSTM Units** | 48 |
| **Hidden Layers** | 2 |
| **Dropout Rate** | 0.3 |
| **Learning Rate** | 0.005 |
| **Activation** | Sigmoid (binary output) |
| **Input Sequence Length** | 20 timesteps |
| **Input Features** | 17 (temporal + maintenance history) |
| **Output** | 1 (failure risk per timestep, range [0, 1]) |
| **Training Epochs** | 3 |
| **Batch Size** | 16 |
| **Early Stopping Patience** | 1 epoch |

---

## 🌳 XGBoost Configuration

| Parameter | Value |
|-----------|-------|
| **Number of Trees** | 250 |
| **Max Tree Depth** | 4 |
| **Learning Rate** | 0.03 |
| **Subsample** | 0.95 |
| **Colsample by Tree** | 0.85 |
| **Min Child Weight** | 2 |
| **Gamma** | 0.05 |
| **Reg Alpha (L1)** | 0.1 |
| **Reg Lambda (L2)** | 3.0 |
| **Tree Method** | Histogram |
| **Input Features** | 20 (17 raw + 3 BiLSTM stack) |
| **Task** | Binary Classification |
| **Objective** | Logistic loss |

---

## 🔧 Feature Engineering (20 Total Features)

### Raw/Temporal Features (17)
1. `asset_type_code` - Equipment type identifier
2. `lab_code` - Laboratory identifier
3. `shift_code` - Work shift code
4. `hour_of_day` - Hour (0-23)
5. `day_of_week` - Day (0-6, Monday-Sunday)
6. `month_number` - Month (1-12)
7. `days_since_last_maintenance` - Days elapsed
8. `rolling_failure_count_7` - Failures in 7-day window
9. `cumulative_failure_count` - Total failures to date
10. `problem_word_count` - Problem description length
11. `solution_word_count` - Solution description length
12. `has_remarks` - Presence of remarks
13. `health_delta` - Health change rate
14. `health_rolling_mean_5` - 5-day average health
15. `health_rolling_std_5` - 5-day health volatility
16. `perf_delta` - Performance change rate
17. `perf_rolling_mean_5` - 5-day average performance

### BiLSTM Stack Features (3)
18. `bilstm_sequence_score` - BiLSTM failure probability (0-1)
19. `bilstm_context_ready` - Context availability flag (0 or 1)
20. `bilstm_failure_risk` - Aligned sequence output

---

## 📋 Model Variants Available

### Hybrid-V1 (Current Active) - Balanced
- **Strengths:** Moderate recall/precision balance
- **BiLSTM:** 48 units, 2 layers
- **Window:** 20-step sequences
- **Threshold Strategy:** Cost-optimized
- **Use Case:** General-purpose failure prediction

### Hybrid-V2 - Precision-Optimized
- **Strengths:** Minimize false alarms
- **BiLSTM:** 64 units, 2 layers
- **Window:** 24-step sequences
- **Max FP Rate:** 4% (tighter control)
- **Use Case:** High-confidence predictions preferred

### Hybrid-V3 - Recall-Optimized
- **Strengths:** Catch all failures (minimize missed failures)
- **BiLSTM:** 40 units, 1 layer
- **Window:** 16-step sequences
- **Max FP Rate:** 8% (loose - accepts more false positives)
- **Use Case:** Safety-critical applications

---

## 💾 Model Artifacts

| File | Size | Last Modified | Status |
|------|------|---------------|--------|
| `models/bilstm.h5` | 1.03 MB | May 2, 04:38 | ✅ Ready |
| `models/xgb_model.json` | 510 KB | May 2, 04:38 | ✅ Ready |
| `models/xgb_metadata.json` | Metadata | May 2, 04:38 | ✅ Ready |

---

## 🚀 Prediction Pipeline Summary

**Three Hybrid Comparison Modes:**

| Mode | Formula | Focus |
|------|---------|-------|
| **V1** | 0.80 × XGB + 0.20 × BiLSTM | Balanced |
| **V2** | XGB(features + BiLSTM_score) | Stacked (strongest) |
| **V3** | 0.60 × Stacked + 0.40 × BiLSTM | Recall-leaning |

**Runtime Scoring:**
- Receives 17 engineered features
- BiLSTM generates 20-step sequence scores
- Aligns to row level
- XGBoost combines signals
- Platt calibration applied
- Final output: Risk probability [0, 1]

---

## ✅ Implementation Checklist

- ✅ Objective 1: Data preprocessing pipeline complete
- ✅ Objective 2: BiLSTM model trained and saved
- ✅ Objective 3: XGBoost stacking layer trained
- ✅ Objective 4: Three variants configured (V1, V2, V3)
- ✅ Objective 5: All metrics calculated
  - ✅ Accuracy
  - ✅ Precision
  - ✅ Recall
  - ✅ F1-Score
  - ✅ PR-AUC
  - ✅ Balanced Accuracy
  - ✅ False Positive Rate

---

## 📌 Key Findings

1. **High Recall (91.74%):** Model catches most failures
2. **Low False Positive Rate (2.68%):** Minimal false alarms
3. **Strong Calibration:** Platt scaling improves probability reliability
4. **Temporal Capture:** BiLSTM layer sequences temporal dependencies
5. **Feature Synergy:** 20-feature stacking leverages both temporal + structured data
6. **Imbalance Handling:** Scale position weight (6.79) properly weights minority class

---

## 🔍 Interpretation Guide

- **Accuracy 97.73%:** Out of every 100 predictions, 97 are correct
- **Precision 84.45%:** When model predicts "failure", it's right 84% of time
- **Recall 91.74%:** Model finds 92 out of 100 actual failures
- **F1 87.94%:** Good balance between precision and recall
- **FP Rate 2.68%:** Only 2-3 false alarms per 100 predictions

This indicates a well-tuned model suitable for predictive maintenance with strong failure detection while maintaining acceptable false alarm rates.
