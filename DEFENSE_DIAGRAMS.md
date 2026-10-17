# Defense Diagrams

Use these Mermaid diagrams in slides, reports, or the thesis defense deck.

## 0) Thesis Context Diagram

```mermaid
flowchart LR
    T[Equipment Maintenance Technician]
    M[Maintenance Manager / Administrator]
    R[Data Scientist / Researcher]
    DS[(Telemetry and Maintenance Logs\nCSV / sensor streams / incident reports)]
    DB[(Operational Database)]
    MA[(Trained Model Artifacts\nBiLSTM + XGBoost + metadata)]

    subgraph SYS[Predictive Maintenance Hybrid System\nBiLSTM + XGBoost Dashboard]
        API[Flask Backend API]
        ENG[Hybrid Inference Engine\nTemporal + Structured Classification]
        UI[Web Dashboard]
    end

    DS --> API
    API <--> DB
    MA --> ENG
    API --> ENG
    ENG --> API
    API --> UI

    T <--> UI
    M <--> UI
    R --> API
    API --> R
```

## 1) System Architecture

```mermaid
flowchart LR
    A[Raw monitoring data\nCSV / sensor logs / incident reports] --> B[Data loading\nload_csv + normalization]
    B --> C[Schema processing\nmonitoring_schema.py]
    C --> D[Feature selection\nraw + temporal + history features]
    D --> E[BiLSTM sequence model\nfirst-stage risk scoring]
    E --> F[Sequence score alignment]
    F --> G[XGBoost final model\nstacked classifier]
    G --> H[Calibration + thresholding]
    H --> I[Failure risk prediction]
    I --> J[Flask dashboard]
    I --> K[Saved artifacts\nmodels/ + config/]
```

## 2) Training Pipeline

```mermaid
flowchart TD
    A[Training CSV] --> B[Normalize schema]
    B --> C{Dataset type?}
    C -->|Lab maintenance| D[Derive will_fail_24h\nfrom Problem column]
    C -->|Synthetic telemetry| E[Use raw telemetry features]
    D --> F[Chronological split]
    E --> F
    F --> G[Train BiLSTM on windows]
    G --> H[Predict sequence probabilities]
    H --> I[Align probabilities to rows]
    I --> J[Attach stacked BiLSTM features]
    J --> K[Train XGBoost]
    K --> L[Probability calibration]
    L --> M[Threshold selection]
    M --> N[Save model artifacts]
```

## 3) Inference Flow

```mermaid
flowchart LR
    A[Incoming records] --> B[Normalize monitoring schema]
    B --> C[Load XGBoost model]
    C --> D{Stacked model enabled?}
    D -->|Yes| E[Load BiLSTM model]
    E --> F[Create sliding windows]
    F --> G[BiLSTM probability scores]
    G --> H[Attach stack features]
    D -->|No| I[Use raw feature set]
    H --> J[XGBoost predict_proba]
    I --> J
    J --> K[Apply calibration]
    K --> L[Compare against threshold]
    L --> M[Failure risk label]
    M --> N[Dashboard / API response]
```

## 4) Dashboard Deployment

```mermaid
sequenceDiagram
    participant U as User
    participant F as Flask app
    participant D as Data loader
    participant M as ML models
    participant DB as Database

    U->>F: Open dashboard / submit report
    F->>D: Load and normalize data
    D-->>F: Prepared dataframe
    F->>M: Load BiLSTM + XGBoost artifacts
    M-->>F: Predictions and risk scores
    F->>DB: Store / update records
    F-->>U: Render dashboard view
```

## 5) Model Interaction View

```mermaid
flowchart TB
    subgraph Sequence_Model[BiLSTM Sequence Model]
        S1[Windowed input sequence] --> S2[Bidirectional LSTM layers]
        S2 --> S3[Dropout]
        S3 --> S4[Sigmoid output]
    end

    subgraph Decision_Model[XGBoost Decision Model]
        X1[Raw telemetry + stack features] --> X2[Tree ensemble]
        X2 --> X3[Class probability]
        X3 --> X4[Thresholded decision]
    end

    S4 --> X1
```

## Notes for the defense

- The BiLSTM is the sequence scorer and the XGBoost model is the final decision layer.
- The stack feature is only available when the model is trained or deployed in stacked mode.
- The dashboard is a Flask app that loads the saved artifacts and displays the resulting risk scores.