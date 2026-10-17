# Research Methodology Framework
## Performance of Temporal-Classification of Hybrid Algorithm for Predicting Failure Maintained Equipment

---

## ANALYSIS PHASE

### T7. Defining Problems
During the literature review and data-gathering process, the researchers identified several critical challenges in equipment failure prediction:

**Problems Identified:**
1. **Temporal Pattern Recognition Gap** - Traditional statistical models fail to capture sequential dependencies in equipment sensor data over time
2. **Feature Complexity** - Equipment failures involve both temporal sequences (health trends, degradation patterns) and structured features (maintenance history, equipment type, usage intensity)
3. **Limited Hybrid Approaches** - Existing models either specialize in temporal analysis (LSTM) or feature-based classification (tree-based) but rarely combine both effectively
4. **Class Imbalance** - Actual failures represent only 12% of operational records, making standard classification problematic
5. **Probability Calibration** - Raw model outputs lack reliability for decision-making

**Proposed Solution:** BiLSTM–XGBoost Hybrid Architecture
- First-stage BiLSTM captures temporal degradation patterns (20-step sequences)
- Second-stage XGBoost performs structured feature classification with BiLSTM outputs as augmented features
- Stacked architecture enables synergistic learning: temporal context + structured inference

---

### T8. Proposal Making and Title Defense
The researchers submitted their research proposal to the panel:

**Proposal Title:**  
*"Performance of Temporal-Classification of Hybrid Algorithm for Predicting Failure Maintained Equipment"*

**Proposal Components:**
- **Background:** Equipment failure is costly; predictive maintenance reduces downtime and maintenance costs
- **Objectives:**
  1. Collect, clean, and preprocess log-based records of asset usage, maintenance, and incidents
  2. Develop a hybrid BiLSTM–XGBoost model analyzing temporal patterns and structured features
  3. Develop a stacked BiLSTM–XGBoost hybrid model capturing both temporal and structured data
  4. Configure and compare multiple parameterized versions (V1, V2, V3)
  5. Evaluate performance using accuracy, precision, recall, and F1-score
  
- **Scope:**
  - Equipment maintenance dataset: 143,592 records with 17 engineered features
  - Training method: Chronological time-series split (80% train, 10% validation, 10% test)
  - Model variants: Three configurations for different optimization strategies
  - Performance benchmarking: Comparative analysis across metrics and holdout test sets

**Panel Review & Approval:** After receiving comments on theoretical justification and implementation feasibility, the proposal was refined and approved as the official study title.

---

### T9. User Requirement Analysis
User requirements were analyzed to determine what stakeholders need from the predictive maintenance system:

**System Users & Requirements:**

**Equipment Maintenance Technicians:**
- View failure risk predictions for monitored assets
- Receive clear alerts when failure probability exceeds thresholds
- Access historical maintenance records and trend analysis
- Understand model confidence and prediction reasoning

**Maintenance Managers/Administrators:**
- Monitor all equipment assets in real-time
- Plan preventive maintenance schedules based on predictions
- Evaluate trade-offs between false alarms and missed failures
- Configure alert thresholds based on business priorities
- Generate comparative reports on model performance

**Data Scientists/Researchers:**
- Train and validate multiple model variants (V1, V2, V3)
- Compare variant performance across metrics (precision, recall, F1)
- Adjust hyperparameters and threshold strategies
- Export predictions and metrics for analysis

**Functional Requirements Derived:**
- Real-time prediction pipeline capable of processing new equipment data
- Three configurable model modes (V1: balanced, V2: precision-optimized, V3: recall-optimized)
- Comprehensive metrics calculation (accuracy, precision, recall, F1-score, PR-AUC)
- Probability calibration for reliable confidence scores
- Database storage for predictions and maintenance history
- Dashboard visualization of predictions across assets

These requirements guided the architecture and feature selection for the hybrid model system.

---

## DESIGN PHASE

### T10. System Logical and Physical Design

#### **Logical System Design**
The logical architecture describes how the system processes data and generates predictions:

```
Raw Sensor/Maintenance Data
        ↓
[Data Normalization & Feature Engineering]
  - Temporal encoding (hour, day, month)
  - Rolling statistics (3, 5, 7-day windows)
  - Cumulative aggregations (failure counts, maintenance history)
  - Result: 17 engineered features per record
        ↓
[Sequence Windowing]
  - Create overlapping 20-step time windows
  - Maintain chronological integrity
  - Result: (N-19, 20, 17) shaped sequences
        ↓
[BiLSTM Temporal Analysis]
  - 2 Bidirectional LSTM layers (48 units)
  - Output: Row-aligned failure risk scores
  - Dropout: 0.3 (regularization)
        ↓
[Feature Augmentation]
  - Attach BiLSTM outputs to raw features
  - Total augmented features: 20 (17 raw + 3 BiLSTM)
        ↓
[XGBoost Structured Classification]
  - 250 trees, max_depth=4
  - Inputs: Augmented 20-feature set
  - Output: Binary failure probability (0-1)
        ↓
[Probability Calibration - Platt Scaling]
  - Improve probability reliability
  - Brier score reduction: 0.0454
        ↓
Final Prediction with Confidence Score
```

#### **Physical System Design**

**Backend Architecture:**
- **Framework:** Flask (Python REST API)
- **Models Storage:** 
  - `models/bilstm.h5` - Trained Keras BiLSTM model
  - `models/xgb_model.json` - Trained XGBoost classifier
  - `models/xgb_metadata.json` - Configuration and metrics
- **Database Schema:**
  - Equipment table: asset metadata
  - SensorReading table: time-series telemetry
  - MaintenanceRecord table: maintenance events
  - Prediction table: model outputs and scores

**Frontend Architecture:**
- **Dashboard:** HTML/CSS/JavaScript interface
- **Views:**
  - Real-time equipment status
  - Asset-level risk predictions
  - Comparative model variant analysis
  - Historical trend visualization

**Data Flow:**
```
CSV Data Input → Data Loader → Schema Normalizer → Feature Engineer
        ↓
Time Series Windows → BiLSTM Model → Sequence Scores → Feature Augmentation
        ↓
XGBoost Classifier → Raw Probabilities → Calibration → Calibrated Scores
        ↓
Threshold Classification → Prediction Result → Dashboard Visualization
```

**Interface Layout:**
- Main dashboard showing asset status grid
- Individual asset prediction detail view
- Model variant comparison panel
- Metrics display (accuracy, precision, recall, F1)
- Alert threshold configuration
- Maintenance history timeline

---

## IMPLEMENTATION PHASE

### T11. Model Development and Training
The researchers implemented the hybrid BiLSTM–XGBoost system according to design specifications:

**Implementation Components:**

1. **Data Pipeline** (`src/data_loader.py`, `src/monitoring_schema.py`)
   - CSV loading and pandas processing
   - Schema detection (lab maintenance vs. synthetic vs. raw IoT)
   - Column normalization and feature engineering
   - Chronological train/test/validation splitting

2. **BiLSTM Model Implementation** (`src/models.py`)
   ```python
   - Input: (batch_size, 20 timesteps, 17 features)
   - Layer 1: Bidirectional LSTM(48 units, return_sequences=True) + Dropout(0.3)
   - Layer 2: Bidirectional LSTM(48 units, return_sequences=False) + Dropout(0.3)
   - Output: Dense(1, activation='sigmoid') → Failure probability
   - Optimizer: Adam(learning_rate=0.005)
   - Loss: Binary Crossentropy
   ```

3. **XGBoost Stacking Implementation** (`src/train.py`)
   ```python
   - Input features: 20 (raw 17 + BiLSTM 3)
   - Configuration:
     * n_estimators: 250
     * max_depth: 4
     * learning_rate: 0.03
     * subsample: 0.95
     * colsample_bytree: 0.85
   - Probability calibration: Platt scaling on validation set
   - Threshold optimization: Cost-based (FN_cost=5, FP_cost=1)
   ```

4. **Three Model Variants Trained:**

   **Hybrid-V1 (Balanced)** - Production Default
   - BiLSTM: 48 units, 2 layers, LR=0.005, window=20
   - XGBoost: depth=4, 250 trees, LR=0.03
   - Threshold strategy: Cost optimization
   - Optimization: Balanced precision-recall

   **Hybrid-V2 (Precision-Optimized)**
   - BiLSTM: 64 units, 2 layers, LR=0.003, window=24
   - XGBoost: depth=5, 320 trees, LR=0.025
   - Threshold strategy: F-Beta (β=2.5)
   - Optimization: Minimize false positives (Max FP Rate: 4%)

   **Hybrid-V3 (Recall-Optimized)**
   - BiLSTM: 40 units, 1 layer, LR=0.007, window=16
   - XGBoost: depth=3, 220 trees, LR=0.04
   - Threshold strategy: Recall maximization
   - Optimization: Minimize false negatives (Max FP Rate: 8%)

5. **Backend API Implementation** (`app.py`)
   - Flask endpoints for predictions
   - Model loading and inference
   - Database integration
   - CORS configuration for frontend access

6. **Prediction Pipeline** (`src/predict_xgb.py`)
   - Load trained models
   - Apply sequence windowing to new data
   - Generate BiLSTM features
   - Stack into XGBoost
   - Apply probability calibration
   - Return risk scores and classifications

---

### T12. Project Presentation
The researchers finalized implementation documents and system configuration:

**Documentation Completed:**
- `README.md` - Project overview and quickstart
- `MODEL_VERIFICATION_REPORT.md` - Objective alignment verification
- `METRICS_DETAILED.md` - Comprehensive metrics breakdown
- `CLASSIFICATION_GUIDE.md` - Model interpretation guide
- `COMPARATIVE_ANALYSIS_API.md` - API endpoint documentation
- Training scripts: `train_v1.ps1`, `train_v2.ps1`, `train_v3.ps1`

**System Configuration Verified:**
- ✅ Data pipeline processes 143,592 training samples
- ✅ BiLSTM model saves/loads correctly (1.03 MB artifact)
- ✅ XGBoost model saves/loads correctly (510 KB artifact)
- ✅ Feature engineering produces 20-dimensional inputs
- ✅ Probability calibration improves reliability (Brier: 0.0454)
- ✅ All three model variants trained and ready for comparison

**Performance Metrics Confirmed:**
- Accuracy: 97.73%
- Precision: 84.45%
- Recall: 91.74%
- F1-Score: 87.94%
- PR-AUC: 94.31%

---

### T13. Delivery and Support
The researchers prepared deployment infrastructure and support mechanisms:

**Delivery Components:**
- **Model Artifacts:** BiLSTM and XGBoost models packaged and versioned
- **Training Scripts:** PowerShell automation for retraining model variants
- **API Endpoint:** Flask backend serving predictions in real-time
- **Dashboard:** Web interface for visualization and monitoring
- **Database Schema:** SQLite/MySQL support for data persistence

**Support & Maintenance Protocol:**
- Model retraining pipeline for incorporating new data
- Threshold adjustment procedures for operational needs
- Variant comparison framework for performance evaluation
- Troubleshooting guides for common issues
- Documentation for integrating new equipment types

---

## TESTING AND MAINTENANCE PHASE

### T14. System Testing and Debugging
The researchers evaluated the hybrid model system to ensure efficiency and alignment with objectives:

**Functional Testing:**
- ✅ Data loading pipeline correctly normalizes multiple data formats
- ✅ Feature engineering produces expected 17 temporal/structured features
- ✅ Time-series windowing maintains chronological integrity
- ✅ BiLSTM forward pass generates sequence scores
- ✅ XGBoost stacking layer correctly augments features
- ✅ Probability calibration improves prediction reliability
- ✅ Threshold optimization balances precision/recall appropriately

**Performance Testing - Future Holdout Test Set:**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Accuracy | ≥95% | 97.73% | ✅ Exceeded |
| Precision | ≥80% | 84.45% | ✅ Exceeded |
| Recall | ≥90% | 91.74% | ✅ Exceeded |
| F1-Score | ≥85% | 87.94% | ✅ Exceeded |
| False Positive Rate | ≤5% | 2.68% | ✅ Exceeded |

**Model Variant Comparison Testing:**
- V1 (Balanced): Moderate precision/recall balance
- V2 (Precision-Opt): Superior false positive control
- V3 (Recall-Opt): Maximum failure detection capability
- All variants ready for deployment based on use-case priorities

**Robustness Testing:**
- ✅ Model handles class imbalance (12% positive class)
- ✅ Handles missing features gracefully (fillna strategy)
- ✅ Maintains performance across chronological test split
- ✅ Probability calibration prevents overconfidence

**Debugging & Issue Resolution:**
- TensorFlow/Keras compatibility verified
- XGBoost serialization/deserialization tested
- Data pipeline handles edge cases (empty sequences, missing values)
- Flask API error handling implemented

---

### T15. Evaluation Metrics and Performance Analysis
The researchers evaluated the model using classification metrics focused on failure prediction quality. Since the study scope centers on model behavior, evaluation emphasized prediction correctness and class-level detection performance.

**Core Evaluation Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | $(TP + TN) / (TP + TN + FP + FN)$ | Overall proportion of correct predictions |
| Precision | $TP / (TP + FP)$ | Of predicted failures, how many were actual failures |
| Recall | $TP / (TP + FN)$ | Of actual failures, how many were correctly detected |
| F1-Score | $2 \cdot (Precision \cdot Recall) / (Precision + Recall)$ | Balance between precision and recall |

Where:
- $TP$ = True Positives
- $TN$ = True Negatives
- $FP$ = False Positives
- $FN$ = False Negatives

**Reported Test Performance (Future Holdout Set):**
- ✅ Accuracy: 97.73%
- ✅ Precision: 84.45%
- ✅ Recall: 91.74%
- ✅ F1-Score: 87.94%

**Evaluation Interpretation:**
- High accuracy confirms strong overall classification reliability.
- High recall shows the model effectively captures most failure events.
- Strong precision indicates false alarms are controlled.
- F1-Score confirms a robust balance for predictive maintenance deployment.

---

### T16. Development of Final Documentation
The researchers completed and finalized all required research documentation:

**Final Documentation Package:**

**Chapter 1: Introduction**
- Problem statement: Equipment failure prediction challenges
- Research objectives aligned with methodology
- Significance of predictive maintenance
- Scope and limitations

**Chapter 2: Review of Related Literature**
- Temporal sequence modeling (LSTM, BiLSTM)
- Tree-based classification methods (XGBoost, Random Forest)
- Hybrid approach justification
- Stacked learning architectures
- Probability calibration techniques
- Equipment failure prediction case studies

**Chapter 3: Technicality of the Study**
- **Data Description:** 
  - Source: Lab maintenance synthetic dataset
  - Volume: 143,592 records, 17 features
  - Time period: Multi-month equipment operations
  
- **Model Architecture:**
  - BiLSTM temporal component (2 layers, 48 units, 20-step sequences)
  - XGBoost stacking layer (250 trees, depth=4)
  - Feature augmentation (20 total features)
  - Probability calibration (Platt scaling)
  
- **Training Configuration:**
  - Chronological split: 80% train, 10% validation, 10% test
  - Optimization: Cost-based threshold selection
  - Class imbalance handling: scale_pos_weight=6.79
  - Early stopping: patience=1 epoch
  
- **Model Variants:**
  - V1 (Balanced): Standard cost optimization
  - V2 (Precision-Opt): 4% FP rate threshold
  - V3 (Recall-Opt): 8% FP rate threshold

**Chapter 4: Methodology**
- **Data Collection:** Equipment maintenance logs and sensor readings
- **Data Preprocessing:** Normalization, feature engineering, windowing
- **Model Development:** BiLSTM architecture, XGBoost stacking
- **Variant Configuration:** Three parameterized versions for comparison
- **Evaluation Framework:** Metrics (accuracy, precision, recall, F1, PR-AUC)
- **Implementation:** Flask backend, dashboard interface

**Chapter 5: Summary and Conclusions**
- **Key Findings:**
  - Hybrid BiLSTM–XGBoost achieves 87.94% F1-score
  - Temporal + structured feature combination outperforms single-method approaches
  - V1 provides balanced production deployment
  - V2/V3 enable use-case-specific optimization
  
- **Contributions:**
  - Stacked hybrid architecture for equipment failure prediction
  - Systematic framework for comparing model variants
  - Probability calibration improving decision reliability
  - Practical system for predictive maintenance scheduling
  
- **Limitations:**
  - Synthetic data used; real-world validation recommended
  - 12% positive class; additional rare-failure testing needed
  - Hyperparameter tuning may vary by equipment type
  
- **Future Work:**
  - Acquire and validate authentic operational maintenance datasets from partner facilities to improve external validity
  - Implement scripted model retraining workflows that can be triggered from the frontend and automatically refresh maintenance dashboards/visuals
  - Deploy continuous retraining with model-drift monitoring to maintain stable predictive performance over time
  - Validate the framework across more facilities, equipment categories, and longer operating horizons for generalizability
  - Add explainability outputs to support maintenance staff interpretation of high-risk predictions
  - Tune thresholds using explicit false-positive vs. false-negative cost trade-offs aligned with organizational priorities
  - Strengthen data governance through stricter timestamp consistency, quality checks, and complete asset history tracking
  - Integrate near-real-time ingestion and alerting for high-criticality maintenance environments
  - Analyze multi-asset failure correlation patterns for coordinated maintenance planning

---

## Research Completion Status

### ✅ All Research Objectives Completed:

1. **Data Collection & Preprocessing:** 143,592 samples, 17 engineered features
2. **BiLSTM Temporal Model:** 2-layer architecture capturing 20-step sequences
3. **XGBoost Stacking:** Hybrid combination of temporal + structured features
4. **Model Variants:** V1, V2, V3 configured for comparative analysis
5. **Performance Evaluation:** All metrics calculated (accuracy, precision, recall, F1-score, PR-AUC)

### ✅ Documentation Complete:
- Introduction, Literature Review, Technical Details, Methodology, Conclusions
- Model verification reports
- Metrics detailed analysis
- Classification interpretation guides
- API documentation
- Deployment scripts

### ✅ System Ready for:
- Production deployment
- Comparative performance analysis
- Real-time failure prediction
- Maintenance scheduling optimization
- Further research validation

---

**Research Completion Date:** May 6, 2026  
**Status:** ✅ COMPLETE AND VALIDATED
