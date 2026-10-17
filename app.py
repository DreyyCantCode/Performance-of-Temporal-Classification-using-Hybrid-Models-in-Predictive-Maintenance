"""Flask application for Performance of Temporal-Classification of Hybrid Algorithm for Predicting Failure Maintained Equipment."""

from __future__ import annotations

import os
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model as keras_load_model

# Import database models and session
from database import Equipment, SessionLocal, IncidentReport, MaintenanceRecord, SensorReading, create_tables, ensure_equipment_schedule_columns

# Import your existing ML modules
from src.monitoring_schema import RAW_MODEL_FEATURE_COLUMNS, normalize_monitoring_schema
from src.data_loader import create_time_series_windows, load_csv
from src.predict_xgb import apply_probability_calibration, attach_bilstm_stack_features, load_xgb_model

# Import comparative analysis module
from src.comparative_analysis import (
    validate_asset_data,
    compare_models_for_asset,
    compare_models_across_assets,
    generate_comparative_report,
)

DEFAULT_FRONTEND_DIR = 'Front End'
LEGACY_FRONTEND_DIR = 'ThesisFrontEnd'
if (Path(__file__).parent / LEGACY_FRONTEND_DIR).exists():
    FRONTEND_DIR_NAME = LEGACY_FRONTEND_DIR
elif (Path(__file__).parent / DEFAULT_FRONTEND_DIR).exists():
    FRONTEND_DIR_NAME = DEFAULT_FRONTEND_DIR
else:
    FRONTEND_DIR_NAME = DEFAULT_FRONTEND_DIR

app = Flask(__name__, static_folder=FRONTEND_DIR_NAME, static_url_path='')
CORS(app)
app.logger.info(f"Using frontend folder: {FRONTEND_DIR_NAME}")

# Configuration
DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'
METRICS_CONFIG_FILE = Path(__file__).parent / 'config' / 'metrics.json'
DEFAULT_SYNTHETIC_DATA_PATH = DATA_DIR / 'lab_maintenance_synthetic_daily.csv'
DEFAULT_FALLBACK_DATA_PATH = DATA_DIR / 'lab_maintenance_full.csv'
GRID_SEARCH_SUMMARY_PATH = MODELS_DIR / 'grid_search' / 'grid_search_summary.json'
XGB_MODEL_PATH = MODELS_DIR / 'xgb_model.json'
BILSTM_MODEL_PATH = MODELS_DIR / 'bilstm.h5'
XGB_METADATA_PATH = MODELS_DIR / 'xgb_metadata.json'
MODEL_FEATURE_COLUMNS = RAW_MODEL_FEATURE_COLUMNS
DEFAULT_FAILURE_HORIZON_HOURS = int(os.getenv('FAILURE_HORIZON_HOURS', '168') or 168)
FAILURE_PROBABILITY_HORIZONS = (24, 72, 168, 336)
REPORT_PRIORITY_WEIGHTS = {
    'low': 0.1,
    'medium': 0.3,
    'high': 0.6,
    'critical': 0.9,
}
MAINTENANCE_TYPE_WEIGHTS = {
    'preventive': -0.08,
    'corrective': -0.05,
    'emergency': -0.02,
}
MAINTENANCE_STATUS_WEIGHTS = {
    'scheduled': 0.03,
    'in_progress': 0.02,
    'completed': -0.03,
}


def normalize_affected_users(value: Any) -> int:
    """Coerce the affected user count to a non-negative integer."""
    try:
        users = int(float(value))
    except (TypeError, ValueError):
        users = 0
    return max(users, 0)


def normalize_score_value(value: Any) -> float | None:
    """Coerce a score to a 0-100 numeric value, or None if invalid."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(score):
        return None
    return float(max(min(score, 100.0), 0.0))


def affected_users_risk_weight(value: Any) -> float:
    """Map affected user counts to a small risk adjustment."""
    users = normalize_affected_users(value)
    if users <= 0:
        return 0.0
    if users <= 5:
        return 0.01
    if users <= 20:
        return 0.03
    if users <= 50:
        return 0.06
    return 0.10

# Global variables for models and data
xgb_model = None
bilstm_model = None
xgb_metadata: dict[str, Any] = {}
metrics_config: dict[str, Any] = {}
current_data = None
latest_uploaded_data_path: Path | None = None
latest_report_training_path: Path | None = None
training_data_path: Path | None = None
prediction_refresh_interval_seconds = int(os.getenv('PREDICTION_REFRESH_INTERVAL_SECONDS', '300') or 300)
prediction_refresh_lock = threading.Lock()
prediction_refresh_started = False
prediction_refresh_cache: dict[str, Any] = {
    'generated_at': None,
    'latest_df': None,
    'payload': None,
}


def normalize_shift_type(value: str | None) -> str:
    """Normalize user input into a supported school shift."""
    normalized = (value or 'am').strip().lower()
    return 'pm' if normalized == 'pm' else 'am'


def calculate_uptime_hours(shift_type: str | None) -> float:
    """Return the expected uptime hours for a school shift."""
    shift = normalize_shift_type(shift_type)
    return 0.0 if shift == 'pm' else 10.0


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records by replacing NaN/Inf with null."""
    safe_df = df.replace([np.inf, -np.inf], np.nan)
    safe_df = safe_df.astype(object).where(pd.notnull(safe_df), None)
    return cast(list[dict[str, Any]], safe_df.to_dict(orient='records'))


def load_metrics_config():
    """Load metrics configuration from JSON."""
    if METRICS_CONFIG_FILE.exists():
        with open(METRICS_CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "metrics": [],
            "prediction_model": {
                "type": "xgboost",
                "target": "will_fail_24h",
                "failure_threshold": 0.5,
                "high_risk_threshold": 0.7,
                "critical_risk_threshold": 0.9
            }
        }


def format_runtime_model_label(runtime_model: str) -> str:
    """Return a display-friendly model label."""
    labels = {
        'xgboost': 'Stacked BiLSTM -> XGBoost',
        'bilstm_to_xgboost_stack': 'Stacked BiLSTM -> XGBoost',
        'hybrid': 'Stacked BiLSTM -> XGBoost',
        'report_overlay': 'Report Overlay',
        'unavailable': 'Unavailable',
    }
    return labels.get(runtime_model, runtime_model.replace('_', ' ').title())


def normalize_lookup_value(value: Any) -> str:
    """Normalize a string for loose matching across report and equipment fields."""
    return ''.join(character for character in str(value or '').lower() if character.isalnum())


def resolve_report_asset_id(db: Any, asset_id: Any = None, component: Any = None, room: Any = None) -> str | None:
    """Resolve the best matching equipment asset for a report."""
    explicit_asset_id = str(asset_id or '').strip()
    if explicit_asset_id:
        matched_equipment = db.query(Equipment).filter(Equipment.asset_id == explicit_asset_id).first()
        if matched_equipment is not None:
            return matched_equipment.asset_id
        return None

    normalized_component = normalize_lookup_value(component)
    normalized_room = normalize_lookup_value(room)

    if not normalized_component and not normalized_room:
        return None

    best_match: tuple[int, Equipment] | None = None
    for equipment in db.query(Equipment).all():
        score = 0
        normalized_asset_id = normalize_lookup_value(equipment.asset_id)
        normalized_name = normalize_lookup_value(getattr(equipment, 'name', None))
        normalized_description = normalize_lookup_value(getattr(equipment, 'description', None))
        normalized_location = normalize_lookup_value(getattr(equipment, 'location', None))

        if normalized_component:
            if normalized_component == normalized_asset_id:
                score += 6
            elif normalized_component in normalized_asset_id or normalized_component in normalized_name:
                score += 4
            elif normalized_component in normalized_description:
                score += 2

        if normalized_room:
            if normalized_room == normalized_location:
                score += 5
            elif normalized_room in normalized_location or normalized_room in normalized_description:
                score += 3

        if score == 0:
            continue

        if best_match is None or score > best_match[0]:
            best_match = (score, equipment)

    return str(best_match[1].asset_id) if best_match is not None else None


def classify_asset_status(failure_risk: float) -> dict[str, Any]:
    """
    Classify asset status based on metric thresholds and ML predictions.
    
    Returns a dict with status, color, priority, and reasons.
    """
    pred_config = metrics_config.get('prediction_model', {})

    risk_high = pred_config.get('high_risk_threshold', 0.7)
    risk_crit = pred_config.get('critical_risk_threshold', 0.9)

    # Collect issues
    issues = []

    # Determine status from failure risk only
    status = "Normal"
    color = "primary"
    priority = 1

    if failure_risk > risk_crit:
        issues.append(f"Failure risk critical ({failure_risk*100:.1f}%)")
        status = "Critical"
        color = "yellow"
        priority = 4

    # Check Danger conditions
    if priority < 4 and failure_risk > risk_high:
        issues.append(f"Failure risk high ({failure_risk*100:.1f}%)")
        status = "Danger"
        color = "danger"
        priority = 3

    # Check Warning conditions
    if priority < 3 and failure_risk > 0.5:
        issues.append(f"Failure risk elevated ({failure_risk*100:.1f}%)")
        status = "Warning"
        color = "warning"
        priority = 2
    
    return {
        "status": status,
        "color": color,
        "priority": priority,
        "issues": issues,
        "risk_status": "Critical" if failure_risk > risk_crit else "High" if failure_risk > risk_high else "Elevated" if failure_risk > 0.5 else "Low"
    }


def build_dashboard_event_overlays(db: Any) -> dict[str, dict[str, Any]]:
    """Build per-asset modifiers from recent incident reports and maintenance records."""
    overlays: dict[str, dict[str, Any]] = {}

    reports = (
        db.query(IncidentReport)
        .filter(IncidentReport.asset_id.isnot(None))
        .order_by(IncidentReport.asset_id.asc(), IncidentReport.timestamp.desc())
        .all()
    )

    for report in reports:
        asset_id = resolve_report_asset_id(
            db,
            getattr(report, 'asset_id', None),
            getattr(report, 'component', None),
            getattr(report, 'room', None),
        )
        if not asset_id:
            continue

        overlay = overlays.setdefault(asset_id, {
            'open_report_count': 0,
            'report_risk': 0.0,
            'maintenance_risk': 0.0,
            'affected_users_risk': 0.0,
            'latest_report_room': None,
            'latest_report_priority': None,
            'latest_report_status': None,
            'latest_report_description': None,
            'latest_report_affected_users': None,
            'latest_maintenance_type': None,
            'latest_maintenance_status': None,
            'latest_maintenance_notes': None,
        })

        report_status = str(getattr(report, 'status', '') or '').strip().lower()
        report_priority = str(getattr(report, 'priority', '') or '').strip().lower()
        report_affected_users = normalize_affected_users(getattr(report, 'affected_users', None))
        if report_status != 'completed':
            overlay['open_report_count'] += 1
            overlay['report_risk'] += REPORT_PRIORITY_WEIGHTS.get(report_priority, 0.04)
            overlay['affected_users_risk'] += affected_users_risk_weight(report_affected_users)

        if overlay['latest_report_priority'] is None:
            overlay['latest_report_room'] = str(getattr(report, 'room', None) or '')
            overlay['latest_report_priority'] = str(getattr(report, 'priority', None) or '')
            overlay['latest_report_status'] = str(getattr(report, 'status', None) or '')
            overlay['latest_report_description'] = str(getattr(report, 'description', None) or '')
            overlay['latest_report_affected_users'] = report_affected_users if report_affected_users > 0 else None

    maintenance_records = (
        db.query(MaintenanceRecord)
        .filter(MaintenanceRecord.asset_id.isnot(None))
        .order_by(MaintenanceRecord.asset_id.asc(), MaintenanceRecord.maintenance_date.desc())
        .all()
    )

    for record in maintenance_records:
        asset_id = str(getattr(record, 'asset_id', '') or '').strip()
        if not asset_id:
            continue

        overlay = overlays.setdefault(asset_id, {
            'open_report_count': 0,
            'report_risk': 0.0,
            'maintenance_risk': 0.0,
            'affected_users_risk': 0.0,
            'latest_report_room': None,
            'latest_report_priority': None,
            'latest_report_status': None,
            'latest_report_description': None,
            'latest_report_affected_users': None,
            'latest_maintenance_type': None,
            'latest_maintenance_status': None,
            'latest_maintenance_notes': None,
        })

        maintenance_status = str(getattr(record, 'status', '') or '').strip().lower()
        maintenance_type = str(getattr(record, 'maintenance_type', '') or '').strip().lower()

        if overlay['latest_maintenance_type'] is None:
            overlay['latest_maintenance_type'] = str(getattr(record, 'maintenance_type', None) or '')
            overlay['latest_maintenance_status'] = str(getattr(record, 'status', None) or '')
            overlay['latest_maintenance_notes'] = str(getattr(record, 'completion_notes', None) or getattr(record, 'description', None) or '')

        overlay['maintenance_risk'] += MAINTENANCE_STATUS_WEIGHTS.get(maintenance_status, 0.0)
        if maintenance_status == 'completed':
            overlay['maintenance_risk'] += MAINTENANCE_TYPE_WEIGHTS.get(maintenance_type, -0.04)

    return overlays


def apply_dashboard_event_overlays(snapshot: pd.DataFrame, db: Any) -> pd.DataFrame:
    """Apply report and maintenance signals to the latest prediction snapshot."""
    if snapshot.empty or 'asset_id' not in snapshot.columns:
        return snapshot

    overlays = build_dashboard_event_overlays(db)
    equipment_rows = db.query(Equipment.asset_id, Equipment.location).all()
    loc_map = {str(asset_id): (location or "") for asset_id, location in equipment_rows} if equipment_rows else {}
    adjusted = snapshot.copy()

    base_failure_risk = adjusted.get('failure_risk', pd.Series(dtype=float))
    adjusted['base_failure_risk'] = pd.to_numeric(base_failure_risk, errors='coerce').fillna(0.0)
    adjusted['event_risk_adjustment'] = 0.0
    adjusted['open_report_count'] = 0
    adjusted['latest_report_priority'] = None
    adjusted['latest_report_status'] = None
    adjusted['latest_report_description'] = None
    adjusted['latest_maintenance_type'] = None
    adjusted['latest_maintenance_status'] = None
    adjusted['latest_maintenance_notes'] = None

    for index, row in adjusted.iterrows():
        asset_id = str(row.get('asset_id', '') or '').strip()
        if not asset_id:
            continue

        overlay = overlays.get(asset_id)
        if not overlay:
            continue

        adjustment = float(overlay['report_risk']) + float(overlay['maintenance_risk']) + float(overlay['affected_users_risk'])
        adjusted.at[index, 'event_risk_adjustment'] = adjustment
        adjusted.at[index, 'open_report_count'] = int(overlay['open_report_count'])
        adjusted.at[index, 'latest_report_room'] = overlay['latest_report_room'] or loc_map.get(asset_id, "")
        adjusted.at[index, 'latest_report_priority'] = overlay['latest_report_priority']
        adjusted.at[index, 'latest_report_status'] = overlay['latest_report_status']
        adjusted.at[index, 'latest_report_description'] = overlay['latest_report_description']
        adjusted.at[index, 'latest_report_affected_users'] = overlay['latest_report_affected_users']
        adjusted.at[index, 'latest_maintenance_type'] = overlay['latest_maintenance_type']
        adjusted.at[index, 'latest_maintenance_status'] = overlay['latest_maintenance_status']
        adjusted.at[index, 'latest_maintenance_notes'] = overlay['latest_maintenance_notes']

    adjusted['failure_risk'] = pd.to_numeric(adjusted['base_failure_risk'], errors='coerce').fillna(0.0)
    adjusted['failure_risk'] = (adjusted['failure_risk'] + pd.to_numeric(adjusted['event_risk_adjustment'], errors='coerce').fillna(0.0)).clip(lower=0.0, upper=1.0)
    prediction_threshold_source = adjusted.get('prediction_threshold', pd.Series(dtype=float))
    threshold_series = pd.to_numeric(prediction_threshold_source, errors='coerce').fillna(float(xgb_metadata.get('threshold', 0.5) or 0.5))
    adjusted['will_fail_predicted'] = (adjusted['failure_risk'] >= threshold_series).astype(int)

    statuses = adjusted['failure_risk'].apply(classify_asset_status)
    adjusted['status'] = statuses.apply(lambda item: item['status'])
    adjusted['status_color'] = statuses.apply(lambda item: item['color'])
    adjusted['status_priority'] = statuses.apply(lambda item: item['priority'])
    adjusted['status_issues'] = statuses.apply(lambda item: item['issues'])
    adjusted['risk_status'] = statuses.apply(lambda item: item['risk_status'])
    return adjusted


def build_report_overlay_rows(latest_asset_ids: set[str]) -> list[dict[str, Any]]:
    """Build synthetic prediction rows for report-only assets not present in telemetry data."""
    db = SessionLocal()
    try:
        overlays = build_dashboard_event_overlays(db)
        equipment_rows = db.query(Equipment.asset_id, Equipment.location).all()
        loc_map = {str(asset_id): (location or "") for asset_id, location in equipment_rows} if equipment_rows else {}
    finally:
        db.close()

    synthetic_rows: list[dict[str, Any]] = []
    threshold = float(xgb_metadata.get('threshold', 0.5) or 0.5)

    for asset_id, overlay in overlays.items():
        if asset_id in latest_asset_ids:
            continue

        report_based_risk = float(overlay['report_risk']) + float(overlay['affected_users_risk'])
        maintenance_bonus = max(float(overlay['maintenance_risk']), 0.0)
        combined_risk = report_based_risk + maintenance_bonus
        if combined_risk <= 0.0:
            continue

        failure_risk = float(min(max(combined_risk, 0.0), 1.0))
        classification = classify_asset_status(failure_risk)
        asset_room = str(overlay['latest_report_room'] or loc_map.get(str(asset_id), "") or "")

        synthetic_rows.append({
            'timestamp': datetime.now().isoformat(),
            'asset_id': asset_id,
            'room': asset_room,
            'open_report_count': int(overlay['open_report_count']),
            'latest_report_room': asset_room,
            'latest_report_priority': overlay['latest_report_priority'],
            'latest_report_status': overlay['latest_report_status'],
            'latest_report_description': overlay['latest_report_description'],
            'latest_report_affected_users': overlay['latest_report_affected_users'],
            'latest_maintenance_type': overlay['latest_maintenance_type'],
            'latest_maintenance_status': overlay['latest_maintenance_status'],
            'latest_maintenance_notes': overlay['latest_maintenance_notes'],
            'will_fail_24h': int(failure_risk >= threshold),
            'will_fail_predicted': int(failure_risk >= threshold),
            'will_fail_within_7d': int(failure_risk >= 0.5),
            'failure_risk': failure_risk,
            'failure_probability_24h': failure_risk,
            'failure_probability_72h': convert_probability_to_horizon(failure_risk, 72),
            'failure_probability_168h': convert_probability_to_horizon(failure_risk, 168),
            'failure_probability_336h': convert_probability_to_horizon(failure_risk, 336),
            'prediction_source': 'report_overlay',
            'prediction_threshold': threshold,
            'status': classification['status'],
            'status_color': classification['color'],
            'status_priority': classification['priority'],
            'status_issues': classification['issues'],
            'risk_status': classification['risk_status'],
        })

    return synthetic_rows


def get_cached_prediction_payload(force_refresh: bool = False) -> dict[str, Any] | None:
    """Return the cached latest prediction payload, refreshing it if needed."""
    with prediction_refresh_lock:
        cached_payload = prediction_refresh_cache.get('payload')
        cached_df = prediction_refresh_cache.get('latest_df')

    if force_refresh or cached_payload is None or cached_df is None:
        if not refresh_prediction_cache():
            return None
        with prediction_refresh_lock:
            cached_payload = prediction_refresh_cache.get('payload')

    return cast(dict[str, Any] | None, cached_payload)


def build_latest_prediction_snapshot() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Build the latest dashboard-facing prediction snapshot."""
    if current_data is None:
        load_data()
    if current_data is None:
        return None, None

    # Only enrich the last window_size rows per asset — BiLSTM needs the window
    # to make a prediction, but we don't need to run all 143k rows through it.
    window_size = int(xgb_metadata.get('bilstm_window_size', 20) or 20)
    stacked = bool(xgb_metadata.get('stacked_with_bilstm'))
    tail_n = window_size if stacked else 1
    working_df = (
        current_data.sort_values('timestamp')
        .groupby('asset_id', group_keys=False)
        .tail(tail_n)
        .copy()
    )

    df = enrich_with_predictions(working_df)
    latest = df.sort_values('timestamp').groupby('asset_id').tail(1).copy()

    db = SessionLocal()
    try:
        # Ensure each prediction has a `room` value. If missing, attempt to
        # populate it from the registered Equipment.location for the asset.
        try:
            equipment_rows = db.query(Equipment.asset_id, Equipment.location).all()
            if equipment_rows:
                loc_map = {str(asset_id): (location or "") for asset_id, location in equipment_rows}
                latest['room'] = latest.apply(
                    lambda r: (r.get('room') or loc_map.get(str(r.get('asset_id') or ''), "")),
                    axis=1
                )
        except Exception:
            # Non-fatal: if DB lookup fails, continue without enriching rooms
            pass

        latest = apply_dashboard_event_overlays(latest, db)
        latest_asset_ids = {str(asset_id) for asset_id in latest['asset_id'].astype(str).tolist()}
        synthetic_rows = build_report_overlay_rows(latest_asset_ids)
        if synthetic_rows:
            latest = pd.concat([latest, pd.DataFrame(synthetic_rows)], ignore_index=True, sort=False)
    finally:
        db.close()

    numeric_risk = pd.to_numeric(latest['failure_risk'], errors='coerce').fillna(0.0)
    for horizon in FAILURE_PROBABILITY_HORIZONS:
        latest[f'failure_probability_{horizon}h'] = convert_probability_series_to_horizon(numeric_risk, horizon)

    latest['will_fail_within_7d'] = (latest['failure_probability_168h'] >= 0.5).astype(int)

    prediction_columns = [
        'timestamp',
        'asset_id',
        'room',
        'latest_report_room',
        'open_report_count',
        'latest_report_priority',
        'latest_report_status',
        'latest_report_description',
        'latest_report_affected_users',
        'latest_maintenance_type',
        'latest_maintenance_status',
        'latest_maintenance_notes',
        'will_fail_24h',
        'will_fail_predicted',
        'will_fail_within_7d',
        'failure_risk',
        'failure_probability_24h',
        'failure_probability_72h',
        'failure_probability_168h',
        'failure_probability_336h',
        'prediction_source',
    ]
    available_columns = [column for column in prediction_columns if column in latest.columns]

    predictions_with_status = []
    for record in dataframe_to_json_records(latest[available_columns]):
        risk = float(record.get('failure_risk', 0) or 0)
        classification = classify_asset_status(risk)

        record['status'] = classification['status']
        record['status_color'] = classification['color']
        record['status_priority'] = classification['priority']
        record['status_issues'] = classification['issues']
        record['risk_status'] = classification['risk_status']
        record['model_label'] = format_runtime_model_label(str(record.get('prediction_source') or 'unavailable'))

        predictions_with_status.append(record)

    payload = {
        'timestamp': datetime.now().isoformat(),
        'base_horizon_hours': 24,
        'predictions': predictions_with_status,
    }
    return latest, payload


def refresh_prediction_cache() -> bool:
    """Rebuild the cached prediction snapshot for dashboard reads."""
    latest, payload = build_latest_prediction_snapshot()
    if latest is None or payload is None:
        return False

    with prediction_refresh_lock:
        prediction_refresh_cache['generated_at'] = datetime.now().isoformat()
        prediction_refresh_cache['latest_df'] = latest
        prediction_refresh_cache['payload'] = payload

    return True


def get_cached_prediction_snapshot() -> pd.DataFrame | None:
    """Return the cached latest prediction DataFrame."""
    with prediction_refresh_lock:
        cached_df = prediction_refresh_cache.get('latest_df')

    if cached_df is None:
        if not refresh_prediction_cache():
            return None
        with prediction_refresh_lock:
            cached_df = prediction_refresh_cache.get('latest_df')

    return cast(pd.DataFrame | None, cached_df)


def prediction_refresh_loop() -> None:
    """Background loop that refreshes dashboard predictions on a schedule."""
    interval = max(60, int(prediction_refresh_interval_seconds or 300))
    while True:
        try:
            refresh_prediction_cache()
        except Exception as exc:
            app.logger.warning(f"⚠ Prediction refresh job failed: {exc}")

        stop_event = cast(Any, getattr(app, '_prediction_refresh_stop', None))
        if stop_event is not None and stop_event.wait(interval):
            break


def start_prediction_refresh_job() -> None:
    """Start the scheduled dashboard prediction refresh loop once."""
    global prediction_refresh_started

    if prediction_refresh_started:
        return

    with prediction_refresh_lock:
        if prediction_refresh_started:
            return

        stop_event = getattr(app, '_prediction_refresh_stop', None)
        if stop_event is None:
            stop_event = threading.Event()
            setattr(app, '_prediction_refresh_stop', stop_event)

        prediction_refresh_started = True

        worker = threading.Thread(target=prediction_refresh_loop, daemon=True, name='prediction-refresh-job')
        worker.start()
        setattr(app, '_prediction_refresh_thread', worker)


def normalize_horizon_hours(value: Any, default: int = DEFAULT_FAILURE_HORIZON_HOURS) -> int:
    """Parse and clamp a requested forecast horizon in hours."""
    try:
        hours = int(float(value))
    except (TypeError, ValueError):
        hours = int(default)
    return max(24, min(hours, 24 * 30))


def format_horizon_label(hours: int) -> str:
    """Return a human-friendly horizon label."""
    if hours % 24 == 0:
        days = hours // 24
        return '1 day' if days == 1 else f'{days} days'
    return '1 hour' if hours == 1 else f'{hours} hours'


def convert_probability_to_horizon(probability: Any, target_hours: int, base_hours: int = 24) -> float:
    """Convert a base 24h probability into an N-hour probability using a constant hazard assumption."""
    try:
        numeric_probability = float(probability)
    except (TypeError, ValueError):
        numeric_probability = 0.0

    numeric_probability = min(max(numeric_probability, 0.0), 1.0)
    if target_hours <= 0 or base_hours <= 0:
        return numeric_probability
    if target_hours == base_hours:
        return numeric_probability

    return float(1.0 - ((1.0 - numeric_probability) ** (target_hours / base_hours)))


def convert_probability_series_to_horizon(series: pd.Series, target_hours: int, base_hours: int = 24) -> pd.Series:
    """Vectorized conversion of base 24h probabilities into an N-hour horizon."""
    numeric_series = pd.to_numeric(series, errors='coerce').fillna(0.0).clip(lower=0.0, upper=1.0)
    if target_hours <= 0 or base_hours <= 0 or target_hours == base_hours:
        return numeric_series.astype(float)

    exponent = target_hours / base_hours
    converted = 1.0 - np.power(1.0 - numeric_series, exponent)
    return pd.Series(converted, index=numeric_series.index, dtype=float)


def load_best_grid_search_result() -> dict[str, Any]:
    """Load the best verified result from the capped grid-search summary when available."""
    if not GRID_SEARCH_SUMMARY_PATH.exists():
        return {}

    try:
        summary_payload = json.loads(GRID_SEARCH_SUMMARY_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}

    if not isinstance(summary_payload, list) or not summary_payload:
        return {}

    successful_runs = [item for item in summary_payload if isinstance(item, dict) and int(item.get('return_code', 1)) == 0]
    if not successful_runs:
        return {}

    def _score(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = cast(dict[str, Any], item.get('future_holdout_metrics', {}) or item.get('best_logged_metrics', {}) or {})
        return (
            float(metrics.get('pr_auc', 0.0) or 0.0),
            float(metrics.get('f1', 0.0) or 0.0),
            float(metrics.get('balanced_accuracy', 0.0) or 0.0),
            float(metrics.get('recall', 0.0) or 0.0),
            float(metrics.get('precision', 0.0) or 0.0),
        )

    best_run = max(successful_runs, key=_score)
    candidate = cast(dict[str, Any], best_run.get('candidate', {}) or {})
    metrics = cast(dict[str, Any], best_run.get('future_holdout_metrics', {}) or best_run.get('best_logged_metrics', {}) or {})
    runtime_model = str(best_run.get('final_runtime_model', best_run.get('best_logged_model', 'unknown')))

    return {
        'run': str(candidate.get('name', 'unknown')),
        'run_dir': str(best_run.get('run_dir', '') or ''),
        'model': runtime_model,
        'model_label': format_runtime_model_label(runtime_model),
        'metrics': {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        'config': {
            'bilstm_units': int(candidate.get('bilstm_units', 0) or 0),
            'bilstm_hidden_layers': int(candidate.get('bilstm_hidden_layers', 0) or 0),
            'bilstm_learning_rate': float(candidate.get('bilstm_learning_rate', 0.0) or 0.0),
            'batch_size': int(candidate.get('batch_size', 0) or 0),
            'xgb_learning_rate': float(candidate.get('xgb_learning_rate', 0.0) or 0.0),
        },
    }


def init_models():
    """Initialize ML models on startup."""
    global xgb_model, bilstm_model, xgb_metadata
    runtime_model_dir = MODELS_DIR

    best_run = load_best_grid_search_result()
    best_run_dir = Path(str(best_run.get('run_dir', '') or '').strip())
    if best_run_dir.exists():
        runtime_model_dir = best_run_dir
        app.logger.info(
            "Using best verified grid-search model: %s (%s)",
            best_run.get('run', 'unknown'),
            best_run.get('model_label', 'unknown'),
        )

    try:
        xgb_model = load_xgb_model(str(runtime_model_dir / 'xgb_model.json'))
        app.logger.info("✓ XGBoost model loaded successfully")
    except Exception as e:
        if runtime_model_dir != MODELS_DIR:
            app.logger.warning(f"⚠ Could not load best grid-search XGBoost model: {e}")
            try:
                xgb_model = load_xgb_model(str(MODELS_DIR / 'xgb_model.json'))
                app.logger.info("✓ Fallback XGBoost model loaded successfully")
            except Exception as fallback_error:
                app.logger.warning(f"⚠ Could not load fallback XGBoost model: {fallback_error}")
                xgb_model = None
        else:
            app.logger.warning(f"⚠ Could not load XGBoost model: {e}")
            xgb_model = None

    try:
        bilstm_path = runtime_model_dir / 'bilstm.h5'
        if not bilstm_path.exists() and runtime_model_dir != MODELS_DIR:
            bilstm_path = BILSTM_MODEL_PATH
        bilstm_model = keras_load_model(str(bilstm_path))
        app.logger.info("✓ BiLSTM model loaded successfully")
    except Exception as e:
        app.logger.warning(f"⚠ Could not load BiLSTM model: {e}")
        bilstm_model = None

    metadata_path = runtime_model_dir / 'xgb_metadata.json'
    if not metadata_path.exists() and runtime_model_dir != MODELS_DIR:
        metadata_path = XGB_METADATA_PATH

    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                xgb_metadata = json.load(metadata_file)
        except Exception as e:
            app.logger.warning(f"⚠ Could not load XGBoost metadata: {e}")
            xgb_metadata = {}
    else:
        xgb_metadata = {}


def load_data(csv_path: str | Path | None = None):
    """Load the data CSV file."""
    global current_data
    if csv_path is None:
        csv_path = DEFAULT_SYNTHETIC_DATA_PATH if DEFAULT_SYNTHETIC_DATA_PATH.exists() else DEFAULT_FALLBACK_DATA_PATH
    
    try:
        current_data = normalize_monitoring_schema(pd.read_csv(csv_path))
        return current_data
    except Exception as e:
        app.logger.error(f"Error loading data: {e}")
        return None


def reset_data_source() -> Path:
    """Restore the active dashboard data source to the default dataset."""
    global latest_uploaded_data_path, latest_report_training_path, training_data_path

    latest_uploaded_data_path = None
    latest_report_training_path = None
    training_data_path = None

    default_path = DEFAULT_SYNTHETIC_DATA_PATH if DEFAULT_SYNTHETIC_DATA_PATH.exists() else DEFAULT_FALLBACK_DATA_PATH
    load_data(default_path)
    refresh_prediction_cache()
    return default_path


def _derive_report_failure_label(priority: Any, affected_users: Any, status: Any) -> int:
    """Derive a demo binary failure label from report severity and impact."""
    normalized_priority = str(priority or '').strip().lower()
    users = normalize_affected_users(affected_users)
    normalized_status = str(status or '').strip().lower()

    if normalized_priority in {'critical', 'high'}:
        return 1
    if users >= 5:
        return 1
    if normalized_status not in {'completed', 'done'} and normalized_priority in {'medium', 'low'}:
        return 1
    return 0


def _find_report_maintenance_record(db: Any, asset_id: str | None, report_date: pd.Timestamp, window_hours: int = 24) -> MaintenanceRecord | None:
    """Find a nearby maintenance record for the report asset in the chosen time window."""
    if not asset_id:
        return None

    if pd.isna(report_date):
        report_date = pd.Timestamp.now()

    window_end = report_date + pd.Timedelta(hours=window_hours)
    return (
        db.query(MaintenanceRecord)
        .filter(
            MaintenanceRecord.asset_id == asset_id,
            MaintenanceRecord.maintenance_date >= report_date,
            MaintenanceRecord.maintenance_date <= window_end,
        )
        .order_by(MaintenanceRecord.maintenance_date.asc())
        .first()
    )


def _get_sensor_metrics(db: Any, asset_id: str | None, report_date: pd.Timestamp) -> dict[str, Any]:
    if not asset_id:
        return {'metric_1': None, 'metric_2': None, 'metric_3': None}

    reading = (
        db.query(SensorReading)
        .filter(SensorReading.asset_id == asset_id, SensorReading.timestamp <= report_date)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    if reading is None:
        reading = (
            db.query(SensorReading)
            .filter(SensorReading.asset_id == asset_id, SensorReading.timestamp > report_date)
            .order_by(SensorReading.timestamp.asc())
            .first()
        )

    if reading is None:
        return {'metric_1': None, 'metric_2': None, 'metric_3': None}

    return {
        'metric_1': reading.metric_1,
        'metric_2': reading.metric_2,
        'metric_3': reading.metric_3,
    }


def _get_report_origin_details(db: Any, report: IncidentReport) -> dict[str, Any]:
    report_obj = cast(Any, report)
    report_date = pd.to_datetime(str(getattr(report_obj, 'date', '') or ''), errors='coerce')
    if pd.isna(report_date):
        report_date = pd.Timestamp.now()

    report_asset_id_raw = getattr(report_obj, 'asset_id', None)
    report_asset_id = str(report_asset_id_raw).strip() if report_asset_id_raw is not None else ''

    equipment = None
    if report_asset_id:
        equipment = db.query(Equipment).filter(Equipment.asset_id == report_asset_id).first()

    sensor_metrics = _get_sensor_metrics(db, report_asset_id or None, report_date)
    maintenance_record = _find_report_maintenance_record(db, report_asset_id or None, report_date)
    maintenance_date_value = getattr(maintenance_record, 'maintenance_date', None)

    return {
        'asset_name': equipment.name if equipment is not None else None,
        'asset_location': equipment.location if equipment is not None else None,
        'asset_status': equipment.status if equipment is not None else None,
        'asset_shift_type': equipment.shift_type if equipment is not None else None,
        'asset_uptime_hours': equipment.uptime_hours if equipment is not None else None,
        'metric_1': sensor_metrics['metric_1'],
        'metric_2': sensor_metrics['metric_2'],
        'metric_3': sensor_metrics['metric_3'],
        'maintenance_label': 1 if maintenance_record is not None else 0,
        'matched_maintenance_type': getattr(maintenance_record, 'maintenance_type', None),
        'matched_maintenance_status': getattr(maintenance_record, 'status', None),
        'matched_maintenance_date': maintenance_date_value.isoformat() if maintenance_date_value is not None else None,
    }


def export_incident_reports_to_lab_maintenance_csv(db: Any, reports: list[IncidentReport]) -> pd.DataFrame:
    """Convert incident reports into a demo lab maintenance CSV schema."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        report_obj = cast(Any, report)
        report_affected_users_raw = getattr(report_obj, 'affected_users', None)
        report_affected_users = pd.to_numeric(pd.Series([report_affected_users_raw]), errors='coerce').iloc[0]
        report_affected_users_value = int(report_affected_users) if not pd.isna(report_affected_users) else 0

        report_asset_id_raw = getattr(report_obj, 'asset_id', None)
        report_asset_id = str(report_asset_id_raw).strip() if report_asset_id_raw is not None else ''

        report_date = pd.to_datetime(str(getattr(report_obj, 'date', '') or ''), errors='coerce')
        if pd.isna(report_date):
            report_date = pd.Timestamp.now()

        month_value = report_date.strftime('%b-%y')
        lab_value = str(getattr(report_obj, 'room', '') or 'Unknown').strip() or 'Unknown'
        priority = str(getattr(report_obj, 'priority', '') or '').strip()
        component = str(getattr(report_obj, 'component', '') or '').strip()
        reporter = str(getattr(report_obj, 'reported_by', '') or '').strip()
        status_value = str(getattr(report_obj, 'status', '') or '').strip()
        description_value = str(getattr(report_obj, 'description', '') or '').strip()

        problem_parts = [priority]
        if component:
            problem_parts.append(component)
        if description_value:
            problem_parts.append(description_value[:120])
        if reporter:
            problem_parts.append(f"reported by {reporter}")
        problem_value = ' - '.join([part for part in problem_parts if part])

        solution_text = status_value or ''
        origin = _get_report_origin_details(db, report)
        final_label = origin['maintenance_label'] or _derive_report_failure_label(priority, report_affected_users_value, status_value)

        rows.append({
            'Month': month_value,
            'Lab': lab_value,
            'Problem': problem_value,
            'Solution': solution_text,
            'Component': component,
            'Priority': priority,
            'AffectedUsers': report_affected_users_value,
            'HealthScore': getattr(report_obj, 'health_score', None),
            'PerformanceScore': getattr(report_obj, 'performance_score', None),
            'ReportedBy': reporter,
            'Status': status_value,
            'Description': description_value,
            'asset_id': report_asset_id or str(f"REPORT_{component}_{lab_value}").strip(),
            'asset_name': origin['asset_name'],
            'asset_location': origin['asset_location'],
            'asset_status': origin['asset_status'],
            'asset_shift_type': origin['asset_shift_type'],
            'asset_uptime_hours': origin['asset_uptime_hours'],
            'timestamp': report_date.isoformat(),
            'metric_1': origin['metric_1'],
            'metric_2': origin['metric_2'],
            'metric_3': origin['metric_3'],
            'maintenance_label': origin['maintenance_label'],
            'matched_maintenance_type': origin['matched_maintenance_type'],
            'matched_maintenance_status': origin['matched_maintenance_status'],
            'matched_maintenance_date': origin['matched_maintenance_date'],
            'will_fail_24h': final_label,
        })

    if not rows:
        return pd.DataFrame(columns=[
            'Month', 'Lab', 'Problem', 'Solution', 'Component', 'Priority',
            'AffectedUsers', 'HealthScore', 'PerformanceScore', 'ReportedBy', 'Status', 'Description',
            'asset_id', 'asset_name', 'asset_location', 'asset_status', 'asset_shift_type', 'asset_uptime_hours',
            'timestamp', 'metric_1', 'metric_2', 'metric_3',
            'maintenance_label', 'matched_maintenance_type', 'matched_maintenance_status', 'matched_maintenance_date',
            'will_fail_24h'
        ])

    df = pd.DataFrame(rows)
    label_values = set(df['will_fail_24h'].astype(int).tolist())
    if len(label_values) == 1:
        demo_label = 1 if 0 in label_values else 0
        earliest_timestamp = pd.to_datetime(df['timestamp'], errors='coerce').min()
        if pd.isna(earliest_timestamp):
            earliest_timestamp = pd.Timestamp.now()

        fallback_rows = []
        for idx in range(2):
            fallback_timestamp = earliest_timestamp - pd.Timedelta(days=1 + idx)
            fallback_rows.append(
                {
                    'Month': fallback_timestamp.strftime('%b-%y'),
                    'Lab': 'DemoLab',
                    'Problem': f'Demo fallback label {idx + 1}',
                    'Solution': '',
                    'Component': '',
                    'Priority': '',
                    'AffectedUsers': 0,
                    'ReportedBy': '',
                    'Status': '',
                    'Description': '',
                    'asset_id': f'DEMO_ASSET_{idx + 1}',
                    'asset_name': None,
                    'asset_location': None,
                    'asset_status': None,
                    'asset_shift_type': None,
                    'asset_uptime_hours': None,
                    'timestamp': fallback_timestamp.isoformat(),
                    'metric_1': None,
                    'metric_2': None,
                    'metric_3': None,
                    'maintenance_label': 0,
                    'matched_maintenance_type': None,
                    'matched_maintenance_status': None,
                    'matched_maintenance_date': None,
                    'will_fail_24h': demo_label,
                }
            )

        df = pd.concat([pd.DataFrame(fallback_rows), df], ignore_index=True)
    return df


def enrich_with_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach model-backed prediction columns when the required features exist."""
    enriched = normalize_monitoring_schema(df)
    xgb_threshold = float(xgb_metadata.get('threshold', 0.5) or 0.5)
    hybrid_threshold = float(xgb_metadata.get('hybrid_threshold', xgb_threshold) or xgb_threshold)
    required_feature_columns = [str(column) for column in xgb_metadata.get('feature_columns', MODEL_FEATURE_COLUMNS)]
    bilstm_feature_columns = [str(column) for column in xgb_metadata.get('bilstm_feature_columns', required_feature_columns)]
    use_hybrid = bool(xgb_metadata.get('hybrid_enabled')) and bilstm_model is not None
    stacked_with_bilstm = bool(xgb_metadata.get('stacked_with_bilstm'))

    if stacked_with_bilstm:
        if bilstm_model is not None:
            enriched = attach_bilstm_stack_features(enriched, xgb_metadata, bilstm_model)
        else:
            enriched['bilstm_failure_risk'] = np.nan

    has_model_features = all(column in enriched.columns for column in required_feature_columns)
    if xgb_model is not None and has_model_features:
        features = enriched[required_feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        xgb_probabilities = xgb_model.predict_proba(features)[:, 1].astype(float)

        final_probabilities = xgb_probabilities.copy()
        prediction_source = str(xgb_metadata.get('runtime_model', 'xgboost'))
        prediction_threshold = xgb_threshold
        enriched['xgb_failure_risk'] = xgb_probabilities

        if (not stacked_with_bilstm) and use_hybrid and bilstm_model is not None and all(column in enriched.columns for column in bilstm_feature_columns):
            bilstm_features = enriched[bilstm_feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            window_size = int(xgb_metadata.get('bilstm_window_size', 20) or 20)
            if len(bilstm_features) >= window_size:
                sequence_windows = create_time_series_windows(bilstm_features, window_size)
                bilstm_probabilities = bilstm_model.predict(sequence_windows, verbose=0).astype(float).ravel()
                aligned_bilstm = np.full(len(enriched), np.nan, dtype=float)
                aligned_bilstm[window_size - 1 :] = bilstm_probabilities
                hybrid_weight_xgb = float(xgb_metadata.get('hybrid_weight_xgb', 0.6) or 0.6)
                hybrid_weight_bilstm = float(xgb_metadata.get('hybrid_weight_bilstm', 0.4) or 0.4)
                hybrid_probabilities = xgb_probabilities.copy()
                valid_mask = ~np.isnan(aligned_bilstm)
                hybrid_probabilities[valid_mask] = (
                    (hybrid_weight_xgb * xgb_probabilities[valid_mask])
                    + (hybrid_weight_bilstm * aligned_bilstm[valid_mask])
                )
                enriched['bilstm_failure_risk'] = aligned_bilstm
                final_probabilities = hybrid_probabilities
                prediction_source = 'hybrid'
                prediction_threshold = hybrid_threshold
            else:
                enriched['bilstm_failure_risk'] = np.nan
        elif 'bilstm_failure_risk' not in enriched.columns:
            enriched['bilstm_failure_risk'] = np.nan

        final_probabilities = apply_probability_calibration(final_probabilities, xgb_metadata)
        enriched['failure_risk'] = final_probabilities.astype(float)
        enriched['will_fail_predicted'] = (enriched['failure_risk'] >= prediction_threshold).astype(int)
        enriched['prediction_source'] = prediction_source
        enriched['prediction_threshold'] = prediction_threshold
        return enriched

    enriched['failure_risk'] = None
    enriched['will_fail_predicted'] = None
    enriched['prediction_source'] = 'unavailable'
    enriched['prediction_threshold'] = None

    return enriched


# Initialize on startup
@app.before_request
def startup():
    """Run initialization once."""
    if not getattr(app, '_monitoring_initialized', False):
        # Initialize database tables
        try:
            create_tables()
            ensure_equipment_schedule_columns()
            app.logger.info("✓ Database tables initialized")
        except Exception as e:
            app.logger.warning(f"⚠ Could not initialize database tables: {e}")
        
        # Initialize ML models
        init_models()
        load_data()
        global metrics_config
        metrics_config = load_metrics_config()
        # Build initial prediction cache in background so startup is not blocked
        threading.Thread(target=refresh_prediction_cache, daemon=True, name='initial-cache-build').start()
        start_prediction_refresh_job()
        setattr(app, '_monitoring_initialized', True)


# ============================================================================
# Static Routes
# ============================================================================

@app.route('/')
def serve_dashboard():
    """Serve the main dashboard."""
    return send_from_directory(FRONTEND_DIR_NAME, 'index.html')


@app.route('/styles.css')
def serve_styles():
    """Serve CSS file."""
    return send_from_directory(FRONTEND_DIR_NAME, 'styles.css')


@app.route('/dashboard_api.js')
def serve_dashboard_script():
    """Serve the dashboard JavaScript."""
    if (Path(app.root_path) / FRONTEND_DIR_NAME / 'dashboard_standalone.js').exists():
        return send_from_directory(FRONTEND_DIR_NAME, 'dashboard_standalone.js')
    return send_from_directory(FRONTEND_DIR_NAME, 'script.js')


@app.route('/dashboard.js')
def serve_old_dashboard_script():
    """Serve the current dashboard JavaScript."""
    return serve_dashboard_script()


@app.route('/<path:path>')
def serve_static(path):
    """Serve other static files from the configured front-end directory."""
    try:
        return send_from_directory(FRONTEND_DIR_NAME, path)
    except Exception:
        return jsonify({"error": f"File not found: {path}"}), 404


# ============================================================================
# API Routes - Configuration
# ============================================================================

@app.route('/api/config/metrics', methods=['GET'])
def get_metrics_config():
    """Get metrics configuration with threshold details."""
    # Enhance metrics with threshold breakdown.
    prediction_model = metrics_config.get('prediction_model', {})
    enhanced_metrics = []

    for metric in metrics_config.get('metrics', []):
        enhanced = metric.copy()
        enhanced['failure_threshold'] = prediction_model.get('failure_threshold', 0.5)
        enhanced['high_risk_threshold'] = prediction_model.get('high_risk_threshold', 0.7)
        enhanced['critical_risk_threshold'] = prediction_model.get('critical_risk_threshold', 0.9)
        enhanced_metrics.append(enhanced)

    return jsonify({
        'metrics': enhanced_metrics,
        'prediction_model': prediction_model,
        'status_classification': metrics_config.get('status_classification', {}),
        'dashboard': metrics_config.get('dashboard', {}),
    })


@app.route('/api/config/assets', methods=['GET'])
def get_assets_config():
    """Get assets configuration."""
    if current_data is None:
        load_data()

    if current_data is not None and 'asset_id' in current_data.columns:
        asset_ids = sorted(current_data['asset_id'].dropna().astype(str).unique().tolist())
        return jsonify({
            "assets": [
                {
                    "id": asset_id,
                    "name": asset_id,
                    "description": f"Maintenance history for {asset_id}",
                    "status": "active"
                }
                for asset_id in asset_ids
            ]
        })

    assets_config = {
        "assets": [
            {
                "id": "PERIPHERAL_01",
                "name": "Peripheral 01",
                "description": "Lab Equipment 1",
                "status": "active"
            },
            {
                "id": "ASSET_02",
                "name": "Asset 02",
                "description": "Lab Equipment 2",
                "status": "active"
            },
            {
                "id": "MONITOR_01",
                "name": "Monitor 01",
                "description": "Lab Equipment 3",
                "status": "active"
            },
            {
                "id": "PRINTER_01",
                "name": "Printer 01",
                "description": "Lab Equipment 4",
                "status": "active"
            },
            {
                "id": "ROUTER_01",
                "name": "Router 01",
                "description": "Lab Equipment 5",
                "status": "active"
            }
        ]
    }
    return jsonify(assets_config)


@app.route('/api/equipment', methods=['GET'])
def get_equipment():
    """Get all registered monitored systems."""
    db = SessionLocal()
    try:
        equipment = db.query(Equipment).order_by(Equipment.asset_id.asc()).all()
        equipment_records: list[dict[str, Any]] = []

        for item in equipment:
            created_at = cast(Any, getattr(item, 'created_at', None))
            updated_at = cast(Any, getattr(item, 'updated_at', None))
            uptime_hours = cast(Any, getattr(item, 'uptime_hours', None))

            equipment_records.append({
                "id": item.id,
                "asset_id": item.asset_id,
                "name": item.name,
                "description": item.description,
                "room": item.location,
                "location": item.location,
                "shift_type": item.shift_type,
                "uptime_hours": float(uptime_hours) if uptime_hours is not None else 0.0,
                "status": item.status,
                "created_at": created_at.isoformat() if created_at is not None else None,
                "updated_at": updated_at.isoformat() if updated_at is not None else None,
            })

        return jsonify({
            "count": len(equipment_records),
            "equipment": equipment_records
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/equipment', methods=['POST'])
def create_equipment():
    """
    Register a new monitored system for maintenance.
    
    Request body (JSON):
    {
        "asset_id": "ASSET_001",
        "name": "Asset Name",
        "description": "Asset description",
        "room": "Lab 1",
        "shift_type": "am",
        "status": "active"
    }
    """
    db = SessionLocal()
    try:
        data = request.get_json() or {}
        
        # Validate required fields
        required_fields = ['asset_id', 'name', 'description']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "required_fields": required_fields,
                    "success": False
                }), 400
        
        room = data.get('room') or data.get('location')
        if not room:
            return jsonify({
                "error": "Missing required field: room (or location)",
                "success": False
            }), 400
        
        # Validate asset_id format
        asset_id = str(data['asset_id']).strip()
        if len(asset_id) == 0 or len(asset_id) > 50:
            return jsonify({
                "error": "asset_id must be between 1 and 50 characters",
                "success": False
            }), 400
        
        # Check if asset already exists
        existing = db.query(Equipment).filter(Equipment.asset_id == asset_id).first()
        if existing:
            return jsonify({
                "error": f"Asset ID '{asset_id}' already exists",
                "success": False
            }), 409
        
        # Create new equipment
        new_equipment = Equipment(
            asset_id=asset_id,
            name=str(data['name']).strip(),
            description=str(data['description']).strip(),
            location=str(room).strip(),
            shift_type=normalize_shift_type(data.get('shift_type')),
            uptime_hours=calculate_uptime_hours(data.get('shift_type')),
            status=str(data.get('status', 'active')).strip()
        )
        
        db.add(new_equipment)
        db.commit()
        db.refresh(new_equipment)
        
        # Refresh prediction cache to include new asset
        threading.Thread(target=refresh_prediction_cache, daemon=True).start()
        
        return jsonify({
            "success": True,
            "message": "Equipment registered successfully",
            "equipment": {
                "id": new_equipment.id,
                "asset_id": new_equipment.asset_id,
                "name": new_equipment.name,
                "description": new_equipment.description,
                "room": new_equipment.location,
                "location": new_equipment.location,
                "shift_type": new_equipment.shift_type,
                "uptime_hours": float(cast(Any, new_equipment.uptime_hours)) if cast(Any, new_equipment.uptime_hours) is not None else 0.0,
                "status": new_equipment.status,
                "created_at": new_equipment.created_at.isoformat(),
            }
        }), 201
    except ValueError as e:
        return jsonify({
            "error": f"Invalid value provided: {str(e)}",
            "success": False
        }), 400
    except Exception as e:
        db.rollback()
        app.logger.error(f"Error creating equipment: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
    finally:
        db.close()


# ============================================================================
# API Routes - Data & Predictions
# ============================================================================

@app.route('/api/data/logs', methods=['GET'])
def get_logs():
    """Get all logs with predictions."""
    if current_data is None:
        load_data()
    
    if current_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    df = enrich_with_predictions(current_data)
    
    # Convert to JSON-serializable format
    return jsonify({
        "count": len(df),
        "data": dataframe_to_json_records(df)
    })


@app.route('/api/data/asset/<asset_id>', methods=['GET'])
def get_asset_logs(asset_id):
    """Get logs for a specific asset."""
    if current_data is None:
        load_data()
    if current_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    df = current_data[current_data['asset_id'] == asset_id].copy()
    
    if df.empty:
        return jsonify({"error": f"No data for asset {asset_id}"}), 404
    
    df = enrich_with_predictions(df)
    
    return jsonify({
        "asset_id": asset_id,
        "count": len(df),
        "data": dataframe_to_json_records(df)
    })


@app.route('/api/equipment/<asset_id>', methods=['DELETE'])
def delete_equipment(asset_id):
    """Remove a registered equipment asset and refresh prediction cache."""
    db = SessionLocal()
    try:
        equipment = db.query(Equipment).filter(Equipment.asset_id == asset_id).first()
        if not equipment:
            return jsonify({"error": "Equipment not found"}), 404
        db.delete(equipment)
        db.commit()
        threading.Thread(target=refresh_prediction_cache, daemon=True).start()
        return jsonify({"message": f"Equipment {asset_id} deleted successfully"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/predictions/latest', methods=['GET'])
def get_latest_predictions():
    """Get latest predictions for all assets."""
    payload = get_cached_prediction_payload()
    if payload is None:
        return jsonify({'timestamp': datetime.now().isoformat(), 'base_horizon_hours': 24, 'predictions': []})

    try:
        db = SessionLocal()
        equipment_rows = db.query(Equipment.asset_id, Equipment.location).all()
        loc_map = {str(asset_id).strip(): (location or "") for asset_id, location in equipment_rows}
        for prediction in payload.get('predictions', []):
            if not prediction.get('room'):
                asset_id = str(prediction.get('asset_id') or '').strip()
                if asset_id:
                    prediction['room'] = loc_map.get(asset_id, prediction.get('room') or "")
    except Exception:
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass

    return jsonify(payload)


@app.route('/api/predictions/summary', methods=['GET'])
def get_prediction_summary():
    """Get summary statistics of predictions."""
    if current_data is None:
        load_data()
    if current_data is None:
        return jsonify({"error": "No data loaded"}), 500

    requested_horizon = normalize_horizon_hours(request.args.get('horizon_hours'), DEFAULT_FAILURE_HORIZON_HOURS)
    df = enrich_with_predictions(current_data)
    latest_snapshot = get_cached_prediction_snapshot()
    if latest_snapshot is None:
        return jsonify({"error": "No data loaded"}), 500

    summary = {
        "total_records": len(df),
        "unique_assets": latest_snapshot['asset_id'].nunique(),
        "snapshot_records": len(latest_snapshot),
        "selected_horizon_hours": requested_horizon,
        "selected_horizon_label": format_horizon_label(requested_horizon),
        "timestamp_range": {
            "start": str(df['timestamp'].min()),
            "end": str(df['timestamp'].max())
        },
        "metric_stats": {}
    }

    historical_failure_rate_24h = 0.0
    if 'will_fail_24h' in df.columns:
        historical_failure_rate_24h = float(pd.to_numeric(df['will_fail_24h'], errors='coerce').fillna(0.0).mean())

    snapshot_failure_rate_24h = 0.0
    if 'will_fail_24h' in latest_snapshot.columns:
        snapshot_failure_rate_24h = float(pd.to_numeric(latest_snapshot['will_fail_24h'], errors='coerce').fillna(0.0).mean())

    snapshot_risk = pd.to_numeric(latest_snapshot['failure_risk'], errors='coerce').fillna(0.0)
    adjusted_snapshot_risk = convert_probability_series_to_horizon(snapshot_risk, requested_horizon)

    horizon_values = sorted(set([*FAILURE_PROBABILITY_HORIZONS, requested_horizon]))
    failure_rate_by_horizon = {
        f"{hours}h": float(convert_probability_series_to_horizon(snapshot_risk, hours).mean())
        for hours in horizon_values
    }

    summary["historical_failure_rate_24h"] = historical_failure_rate_24h
    summary["snapshot_failure_rate_24h"] = snapshot_failure_rate_24h
    summary["failure_rate"] = failure_rate_by_horizon.get(
        f"{requested_horizon}h",
        convert_probability_to_horizon(snapshot_failure_rate_24h, requested_horizon),
    )
    summary["failure_rate_by_horizon"] = failure_rate_by_horizon

    summary["failure_risk_stats"] = {
        "mean_risk": float(adjusted_snapshot_risk.mean()),
        "high_risk_count": int((adjusted_snapshot_risk >= 0.5).sum()),
        "failure_count": int((adjusted_snapshot_risk >= 0.8).sum()),
        "prediction_source": str(latest_snapshot['prediction_source'].mode().iloc[0]) if 'prediction_source' in latest_snapshot.columns and not latest_snapshot['prediction_source'].dropna().empty else 'unavailable',
    }

    return jsonify(summary)


# --- Training state ---
_training_state: dict[str, Any] = {'status': 'idle', 'started_at': None, 'finished_at': None, 'error': None}
_training_lock = threading.Lock()


def map_runtime_model_to_train_mode(runtime_model: str) -> str:
    normalized = str(runtime_model or '').strip().lower()
    if normalized in {'bilstm_to_xgboost_stack', 'stacked', 'stacked-only', 'hybrid'}:
        return 'hybrid-v1'
    return 'hybrid-v1'


def _run_training_subprocess(data_path: Path | None = None, model_mode: str | None = None) -> None:
    """Run src.train in a background thread and reload models when done."""
    global training_data_path

    python_exe = Path(__file__).parent / '.venv' / 'Scripts' / 'python.exe'
    if not python_exe.exists():
        python_exe = Path('python')

    selected_data_path = (
        data_path
        or training_data_path
        or latest_report_training_path
        or latest_uploaded_data_path
        or DEFAULT_SYNTHETIC_DATA_PATH
    )

    if selected_data_path.exists() and selected_data_path.name.startswith('incident_reports_training_'):
        try:
            report_df = pd.read_csv(selected_data_path)
            fallback_df = pd.read_csv(DEFAULT_FALLBACK_DATA_PATH)
            combined_df = pd.concat([fallback_df, report_df], ignore_index=True, sort=False)
            combined_df = combined_df.drop_duplicates().reset_index(drop=True)
            combined_path = DATA_DIR / 'combined_training_dataset.csv'
            combined_df.to_csv(combined_path, index=False)
            selected_data_path = combined_path
            app.logger.info(
                "Using combined training dataset for report retrain: %s + %s -> %s",
                selected_data_path.name,
                DEFAULT_FALLBACK_DATA_PATH.name,
                combined_path.name,
            )
        except Exception as exc:
            app.logger.warning(f"Could not build combined retraining dataset: {exc}")

    training_data_path = selected_data_path
    runtime_model = str(xgb_metadata.get('runtime_model', 'xgboost'))
    metadata_model_mode = str(xgb_metadata.get('model_mode', '') or '').strip()
    valid_hybrid_modes = {'hybrid-v1', 'hybrid-v2', 'hybrid-v3'}
    default_mode = metadata_model_mode if metadata_model_mode in valid_hybrid_modes else map_runtime_model_to_train_mode(runtime_model)
    effective_model_mode = model_mode or default_mode

    cmd = [
        str(python_exe), '-m', 'src.train',
        '--data', str(selected_data_path),
        '--target', 'will_fail_24h',
        '--task', 'classification',
        '--model-mode', effective_model_mode,
        '--split-method', 'time',
        '--val-size', '0.1',
        '--epochs', '3',
        '--batch-size', '16',
        '--patience', '1',
        '--output-dir', str(MODELS_DIR),
        '--xgb-max-depth', '4',
        '--xgb-n-estimators', '250',
        '--xgb-learning-rate', '0.03',
        '--max-fp-rate', '0.05',
        '--threshold-strategy', 'cost',
        '--fbeta-beta', '2.0',
        '--fn-cost', '5.0',
        '--fp-cost', '1.0',
        '--calibration', 'platt',
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:] if result.stderr else 'Training failed')

        init_models()
        load_data()
        refresh_prediction_cache()

        with _training_lock:
            _training_state['status'] = 'done'
            _training_state['finished_at'] = datetime.now().isoformat()
            _training_state['error'] = None
    except Exception as exc:
        with _training_lock:
            _training_state['status'] = 'error'
            _training_state['finished_at'] = datetime.now().isoformat()
            _training_state['error'] = str(exc)


@app.route('/api/train', methods=['POST'])
def trigger_training():
    """Start a background model training run. Returns immediately; poll /api/train/status."""
    data = request.get_json(silent=True) or {}
    requested_path = data.get('data_path')
    requested_model_mode = data.get('model_mode')

    training_path = None
    if requested_path:
        candidate_path = Path(requested_path)
        if candidate_path.exists():
            training_path = candidate_path
        else:
            app.logger.warning(f"Requested training path does not exist: {requested_path}")

    training_model_mode = None
    if requested_model_mode:
        requested_model_mode = str(requested_model_mode).strip()
        valid_modes = {'hybrid-v1', 'hybrid-v2', 'hybrid-v3'}
        if requested_model_mode in valid_modes:
            training_model_mode = requested_model_mode
        else:
            app.logger.warning(f"Requested model_mode is invalid: {requested_model_mode}")

    with _training_lock:
        if _training_state['status'] == 'running':
            return jsonify({'status': 'running', 'message': 'Training already in progress'}), 409
        _training_state['status'] = 'running'
        _training_state['started_at'] = datetime.now().isoformat()
        _training_state['finished_at'] = None
        _training_state['error'] = None

    t = threading.Thread(target=_run_training_subprocess, args=(training_path, training_model_mode), daemon=True)
    t.start()
    return jsonify({
        'status': 'running',
        'message': 'Training started',
        'data_path': str(training_path) if training_path else None,
        'model_mode': training_model_mode,
    }), 202


@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Return current training state."""
    with _training_lock:
        return jsonify(dict(_training_state))


@app.route('/api/metrics/evaluate', methods=['POST'])
def evaluate_metrics():
    """
    Evaluate metrics and show how they compare to thresholds.
    
    Request body:
    {
        "failure_risk": 0.45
    }
    """
    data = request.get_json()
    risk = float(data.get('failure_risk', 0) or 0)
    risk = min(max(risk, 0.0), 1.0)

    pred_config = metrics_config.get('prediction_model', {})
    high_threshold = float(pred_config.get('high_risk_threshold', 0.7) or 0.7)
    critical_threshold = float(pred_config.get('critical_risk_threshold', 0.9) or 0.9)
    warning_threshold = 0.5

    if risk > critical_threshold:
        status = 'Critical - Immediate Maintenance'
        color = 'yellow'
    elif risk > high_threshold:
        status = 'High Risk - Prioritize Maintenance'
        color = 'danger'
    elif risk > warning_threshold:
        status = 'Elevated Risk - Monitor Closely'
        color = 'warning'
    else:
        status = 'Low Risk - Normal Operation'
        color = 'primary'

    results = [{
        'name': 'failure_risk',
        'label': 'Failure Risk',
        'value': risk,
        'unit': '',
        'status': status,
        'color': color,
        'scoring': 'lower_is_better',
        'thresholds': {
            'good': warning_threshold,
            'warning': high_threshold,
            'bad': critical_threshold,
            'critical': critical_threshold
        },
        'distances': {
            'to_warning': risk - warning_threshold,
            'to_danger': risk - high_threshold,
            'to_critical': risk - critical_threshold,
            'below_good': warning_threshold - risk
        },
        'percentage': round(risk * 100.0, 1)
    }]
    
    return jsonify({
        'metrics': results,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/classify', methods=['POST'])
def classify_metrics():
    """
    Classify asset status based on provided metrics.
    
    Request body:
    {
        "failure_risk": 0.45
    }
    """
    data = request.get_json()

    risk = float(data.get('failure_risk', 0) or 0)

    classification = classify_asset_status(risk)

    return jsonify({
        "metrics": {
            "failure_risk": risk
        },
        "classification": classification,
        "thresholds": {
            "failure_risk": {
                "warning": 0.5,
                "high": metrics_config.get('prediction_model', {}).get('high_risk_threshold'),
                "critical": metrics_config.get('prediction_model', {}).get('critical_risk_threshold')
            }
        }
    })


# ============================================================================
# API Routes - Data Management (for later)
# ============================================================================

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """
    Upload a new CSV file for predictions.
    
    Validates that the file contains required columns and data before loading.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided", "success": False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file", "success": False}), 400
    
    if not file.filename or not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be CSV format", "success": False}), 400
    
    try:
        global latest_uploaded_data_path, current_data
        
        # Read and validate the file
        temp_df = pd.read_csv(file.stream)
        
        if temp_df.empty:
            return jsonify({
                "error": "CSV file is empty",
                "success": False
            }), 400
        
        # Check for minimum required columns
        required_columns = {'asset_id', 'timestamp'}
        missing_columns = required_columns - set(temp_df.columns)
        
        if missing_columns:
            available = list(temp_df.columns[:10])
            return jsonify({
                "error": f"Missing required columns: {missing_columns}",
                "required_columns": list(required_columns),
                "available_columns": available,
                "success": False
            }), 400
        
        # Check for data issues
        null_assets = temp_df['asset_id'].isna().sum()
        null_timestamps = temp_df['timestamp'].isna().sum()
        
        if null_assets > len(temp_df) * 0.5 or null_timestamps > len(temp_df) * 0.5:
            return jsonify({
                "error": "Too many missing values in required columns",
                "asset_id_nulls": int(null_assets),
                "timestamp_nulls": int(null_timestamps),
                "success": False
            }), 400
        
        # Save the validated file
        filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = DATA_DIR / filename
        file.seek(0)  # Reset file pointer
        file.save(filepath)
        
        # Load and normalize the data
        loaded_data = normalize_monitoring_schema(pd.read_csv(filepath))
        
        if loaded_data is None or loaded_data.empty:
            filepath.unlink()  # Clean up invalid file
            return jsonify({
                "error": "Failed to normalize data from file",
                "success": False
            }), 400
        
        # Update global data and cache
        current_data = loaded_data
        latest_uploaded_data_path = filepath
        refresh_prediction_cache()
        
        # Get summary statistics
        unique_assets = loaded_data['asset_id'].nunique() if 'asset_id' in loaded_data.columns else 0
        timestamp_range = {
            'start': str(loaded_data['timestamp'].min()) if 'timestamp' in loaded_data.columns else None,
            'end': str(loaded_data['timestamp'].max()) if 'timestamp' in loaded_data.columns else None,
        }
        
        return jsonify({
            "success": True,
            "message": "File uploaded and validated successfully",
            "filename": filename,
            "filepath": str(filepath),
            "records": len(loaded_data),
            "unique_assets": int(unique_assets),
            "columns": list(loaded_data.columns),
            "timestamp_range": timestamp_range
        }), 201
        
    except pd.errors.ParserError as e:
        return jsonify({
            "error": f"Invalid CSV format: {str(e)}",
            "success": False
        }), 400
    except Exception as e:
        app.logger.error(f"Error uploading data: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/api/data/reset', methods=['POST'])
def reset_data():
    """Reset the dashboard back to the default data source."""
    try:
        default_path = reset_data_source()
        return jsonify({
            "message": "Dashboard data reset successfully",
            "filename": default_path.name,
            "filepath": str(default_path),
            "records": len(current_data) if current_data is not None else 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    runtime_model = str(xgb_metadata.get('runtime_model', 'xgboost'))
    calibration_info = cast(dict[str, Any], xgb_metadata.get('calibration', {}) or {})
    bilstm_params = cast(dict[str, Any], xgb_metadata.get('bilstm_params', {}) or {})
    xgb_params = cast(dict[str, Any], xgb_metadata.get('xgb_params', {}) or {})
    future_holdout_metrics = cast(dict[str, Any], xgb_metadata.get('future_holdout_metrics', {}) or {})
    best_verified_result = load_best_grid_search_result()

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": xgb_model is not None,
        "xgb_runtime_loaded": xgb_model is not None,
        "xgb_artifact_exists": XGB_MODEL_PATH.exists(),
        "xgb_threshold": float(xgb_metadata.get('threshold', 0.5) or 0.5),
        "bilstm_runtime_loaded": bilstm_model is not None,
        "bilstm_artifact_exists": BILSTM_MODEL_PATH.exists(),
        "runtime_model": runtime_model,
        "model_mode": str(xgb_metadata.get('model_mode', 'hybrid-v1') or 'hybrid-v1'),
        "runtime_model_label": format_runtime_model_label(runtime_model),
        "runtime_architecture": "input_to_bilstm_to_xgboost" if runtime_model == 'bilstm_to_xgboost_stack' else runtime_model,
        "hybrid_weight_xgb": float(xgb_metadata.get('hybrid_weight_xgb', 1.0) or 1.0),
        "hybrid_weight_bilstm": float(xgb_metadata.get('hybrid_weight_bilstm', 0.0) or 0.0),
        "hybrid_threshold": float(xgb_metadata.get('hybrid_threshold', xgb_metadata.get('threshold', 0.5)) or 0.5),
        "threshold_strategy": str(xgb_metadata.get('threshold_strategy', 'balanced')),
        "calibration_method": str(calibration_info.get('method', 'none')),
        "model_feature_columns": [str(column) for column in xgb_metadata.get('feature_columns', MODEL_FEATURE_COLUMNS)],
        "bilstm_params": {
            "units": int(bilstm_params.get('units', 0) or 0),
            "hidden_layers": int(bilstm_params.get('hidden_layers', 0) or 0),
            "learning_rate": float(bilstm_params.get('learning_rate', 0.0) or 0.0),
            "batch_size": int(bilstm_params.get('batch_size', 0) or 0),
        },
        "xgb_learning_rate": float(xgb_params.get('learning_rate', 0.0) or 0.0),
        "future_holdout_metrics": {
            key: float(value)
            for key, value in future_holdout_metrics.items()
            if isinstance(value, (int, float))
        },
        "verified_best_result": best_verified_result,
        "data_loaded": current_data is not None,
        "data_records": len(current_data) if current_data is not None else 0
    })


# ============================================================================
# API Routes - Comparative Analysis for Asset Predictions
# ============================================================================

@app.route('/api/analysis/compare-models/<asset_id>', methods=['GET'])
def compare_models_endpoint(asset_id: str):
    """
    Compare predictions from multiple models for a specific asset.
    
    Query parameters:
        - horizon_hours: Optional forecast horizon (default: 24)
    """
    if current_data is None:
        load_data()
    if current_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    # Validate asset exists
    if asset_id not in current_data['asset_id'].values:
        return jsonify({"error": f"Asset {asset_id} not found in data"}), 404
    
    # Enrich with predictions
    df = enrich_with_predictions(current_data)
    
    # Validate the data
    is_valid, error_msg = validate_asset_data(df, asset_id)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    # Compare models
    try:
        analysis = compare_models_for_asset(df, asset_id, xgb_metadata)
        return jsonify(analysis)
    except Exception as e:
        app.logger.error(f"Error comparing models for asset {asset_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analysis/compare-all', methods=['GET'])
def compare_all_assets_endpoint():
    """
    Compare model predictions across all assets.
    
    Query parameters:
        - limit: Maximum number of assets to analyze (default: 50)
    """
    if current_data is None:
        load_data()
    if current_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    limit = request.args.get('limit', default=50, type=int)
    limit = max(1, min(limit, 500))  # Clamp between 1 and 500
    
    # Enrich with predictions
    df = enrich_with_predictions(current_data)
    
    # Validate the data
    is_valid, error_msg = validate_asset_data(df)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    try:
        analysis = compare_models_across_assets(df, xgb_metadata, limit=limit)
        return jsonify(analysis)
    except Exception as e:
        app.logger.error(f"Error comparing assets: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analysis/report', methods=['GET'])
def generate_report_endpoint():
    """
    Generate a comprehensive comparative analysis report across all models and assets.
    """
    if current_data is None:
        load_data()
    if current_data is None:
        return jsonify({"error": "No data loaded"}), 500
    
    # Enrich with predictions
    df = enrich_with_predictions(current_data)
    
    # Validate the data
    is_valid, error_msg = validate_asset_data(df)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    try:
        report = generate_comparative_report(df, xgb_metadata)
        return jsonify(report)
    except Exception as e:
        app.logger.error(f"Error generating report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analysis/validate', methods=['POST'])
def validate_new_data():
    """
    Validate that new asset data can be used for prediction.
    
    Request body (JSON):
    {
        "asset_id": "ASSET_01",
        "timestamp": "2026-05-03T10:00:00",
        "metrics": {
            "temperature_c": 45.5,
            "uptime_hours": 8.0,
            "workload_intensity": 0.75
        }
    }
    
    Returns:
        Validation result with success status and any error messages
    """
    try:
        data = request.get_json() or {}
        
        asset_id = data.get('asset_id')
        timestamp = data.get('timestamp')
        metrics = data.get('metrics', {})
        
        if not asset_id:
            return jsonify({"error": "Missing asset_id", "valid": False}), 400
        
        if not timestamp:
            return jsonify({"error": "Missing timestamp", "valid": False}), 400
        
        # Validate timestamp format
        try:
            pd.to_datetime(timestamp)
        except Exception:
            return jsonify({"error": "Invalid timestamp format", "valid": False}), 400
        
        # Check required metric columns
        required_metrics = ['temperature_c', 'uptime_hours', 'workload_intensity']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        
        if missing_metrics:
            return jsonify({
                "valid": False,
                "error": f"Missing required metrics: {missing_metrics}",
                "required_metrics": required_metrics
            }), 400
        
        # Validate metric values are numeric
        validation_errors = []
        for metric_name, metric_value in metrics.items():
            try:
                float_val = float(metric_value)
                if not (-100 <= float_val <= 1000):  # Reasonable bounds check
                    validation_errors.append(f"{metric_name} value {metric_value} out of reasonable range")
            except (TypeError, ValueError):
                validation_errors.append(f"{metric_name} value is not numeric")
        
        if validation_errors:
            return jsonify({
                "valid": False,
                "errors": validation_errors
            }), 400
        
        # All validations passed
        return jsonify({
            "valid": True,
            "asset_id": asset_id,
            "timestamp": timestamp,
            "message": "Data is valid and ready for prediction"
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error validating data: {e}")
        return jsonify({"error": str(e), "valid": False}), 500


# ============================================================================
# API Routes - Incident Reports
# ============================================================================

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all incident reports."""
    db = SessionLocal()
    try:
        reports = db.query(IncidentReport).order_by(IncidentReport.timestamp.desc()).all()
        report_items = []

        for report in reports:
            origin = _get_report_origin_details(db, report)
            heuristic_label = _derive_report_failure_label(report.priority, report.affected_users, report.status)
            will_fail = origin['maintenance_label'] or heuristic_label

            report_items.append({
                "id": report.id,
                "asset_id": report.asset_id,
                "asset_name": origin['asset_name'],
                "asset_location": origin['asset_location'],
                "asset_status": origin['asset_status'],
                "asset_shift_type": origin['asset_shift_type'],
                "asset_uptime_hours": origin['asset_uptime_hours'],
                "component": report.component,
                "room": report.room,
                "reportedBy": report.reported_by,
                "date": report.date,
                "priority": report.priority,
                "affected_users": report.affected_users,
                "description": report.description,
                "status": report.status,
                "completedOn": report.completed_on,
                "timestamp": report.timestamp,
                "health_score": report.health_score,
                "performance_score": report.performance_score,
                "metric_1": origin['metric_1'],
                "metric_2": origin['metric_2'],
                "metric_3": origin['metric_3'],
                "maintenance_label": origin['maintenance_label'],
                "matched_maintenance_type": origin['matched_maintenance_type'],
                "matched_maintenance_status": origin['matched_maintenance_status'],
                "matched_maintenance_date": origin['matched_maintenance_date'],
                "will_fail_24h": will_fail,
            })

        return jsonify({
            "count": len(reports),
            "reports": report_items
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/reports', methods=['POST'])
def create_report():
    """Create a new incident report."""
    db = SessionLocal()
    try:
        data = request.get_json() or {}
        
        # Validate required fields
        required_fields = ['component', 'room', 'reportedBy', 'date', 'priority', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        asset_id = str(data.get('asset_id', '')).strip() or None
        asset_name = str(data.get('asset_name', '')).strip() or None

        if asset_id:
            linked_asset = db.query(Equipment).filter(Equipment.asset_id == asset_id).first()
            if linked_asset is None:
                return jsonify({"error": "Selected asset does not exist"}), 400
        elif asset_name:
            linked_asset = db.query(Equipment).filter(Equipment.name == asset_name).first()
            if linked_asset is None:
                return jsonify({"error": "Selected asset name does not exist"}), 400
            asset_id = linked_asset.asset_id
        else:
            linked_asset = None
            asset_id = resolve_report_asset_id(db, None, data.get('component'), data.get('room'))
            if not asset_id:
                # Create a synthetic asset_id for reports without linked equipment
                asset_id = f"REPORT_{data['component']}_{data['room']}_{int(datetime.now().timestamp())}".replace(" ", "_")

        linked_asset_name = linked_asset.name if linked_asset is not None else None
        
        # Create new report
        new_report = IncidentReport(
            asset_id=asset_id,
            component=data['component'],
            room=data['room'],
            reported_by=data['reportedBy'],
            date=data['date'],
            priority=data['priority'],
            affected_users=normalize_affected_users(data.get('affected_users')),
            description=data['description'],
            status=data.get('status', 'In Progress'),
            completed_on=data.get('completedOn', ''),
            health_score=normalize_score_value(data.get('health_score')),
            performance_score=normalize_score_value(data.get('performance_score')),
            timestamp=data.get('timestamp', datetime.now().timestamp() * 1000)
        )
        
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        refresh_prediction_cache()

        origin = _get_report_origin_details(db, new_report)
        will_fail_label = origin['maintenance_label'] or _derive_report_failure_label(new_report.priority, new_report.affected_users, new_report.status)
        
        return jsonify({
            "message": "Report created successfully",
            "report": {
                "id": new_report.id,
                "asset_id": new_report.asset_id,
                "component": new_report.component,
                "room": new_report.room,
                "reportedBy": new_report.reported_by,
                "date": new_report.date,
                "priority": new_report.priority,
                "affected_users": new_report.affected_users,
                "description": new_report.description,
                "status": new_report.status,
                "completedOn": new_report.completed_on,
                "timestamp": new_report.timestamp,
                "asset_name": origin['asset_name'] or linked_asset_name,
                "asset_location": origin['asset_location'],
                "asset_status": origin['asset_status'],
                "asset_shift_type": origin['asset_shift_type'],
                "asset_uptime_hours": origin['asset_uptime_hours'],
                "metric_1": origin['metric_1'],
                "metric_2": origin['metric_2'],
                "metric_3": origin['metric_3'],
                "health_score": new_report.health_score,
                "performance_score": new_report.performance_score,
                "maintenance_label": origin['maintenance_label'],
                "matched_maintenance_type": origin['matched_maintenance_type'],
                "matched_maintenance_status": origin['matched_maintenance_status'],
                "matched_maintenance_date": origin['matched_maintenance_date'],
                "will_fail_24h": will_fail_label,
            }
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/reports/<int:report_id>', methods=['PUT'])
def update_report(report_id):
    """Update an incident report."""
    db = SessionLocal()
    try:
        report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
        
        if not report:
            return jsonify({"error": "Report not found"}), 404
        
        data = request.get_json() or {}
        report_record = cast(Any, report)
        
        # Update fields if provided
        if 'component' in data:
            report.component = data['component']
        if 'room' in data:
            report.room = data['room']
        if 'reportedBy' in data:
            report.reported_by = data['reportedBy']
        if 'date' in data:
            report.date = data['date']
        if 'priority' in data:
            report.priority = data['priority']
        if 'affected_users' in data:
            report_record.affected_users = normalize_affected_users(data['affected_users'])
        if 'description' in data:
            report_record.description = data['description']
        if 'status' in data:
            report_record.status = data['status']
        if 'asset_id' in data:
            asset_id_value = str(data.get('asset_id', '') or '').strip()
            if asset_id_value:
                linked_asset = db.query(Equipment).filter(Equipment.asset_id == asset_id_value).first()
                if linked_asset is None:
                    return jsonify({"error": "Selected asset does not exist"}), 400
                report_record.asset_id = asset_id_value
            else:
                report_record.asset_id = None
        if 'completedOn' in data:
            report_record.completed_on = data['completedOn']
        if 'health_score' in data:
            report_record.health_score = normalize_score_value(data.get('health_score'))
        if 'performance_score' in data:
            report_record.performance_score = normalize_score_value(data.get('performance_score'))
        
        report_record.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(report)
        
        return jsonify({
            "message": "Report updated successfully",
            "report": {
                "id": report.id,
                "component": report.component,
                "room": report.room,
                "reportedBy": report.reported_by,
                "date": report.date,
                "priority": report.priority,
                "affected_users": report.affected_users,
                "description": report.description,
                "status": report.status,
                "completedOn": report.completed_on,
                "health_score": report.health_score,
                "performance_score": report.performance_score,
                "timestamp": report.timestamp
            }
        })
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/reports/<int:report_id>', methods=['DELETE'])
def delete_report(report_id):
    """Delete an incident report."""
    db = SessionLocal()
    try:
        report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
        
        if not report:
            return jsonify({"error": "Report not found"}), 404
        
        db.delete(report)
        db.commit()
        threading.Thread(target=refresh_prediction_cache, daemon=True).start()

        return jsonify({"message": "Report deleted successfully"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/reports/export-training-csv', methods=['GET', 'POST'])
def export_reports_training_csv():
    """Export incident reports to a demo training-compatible CSV."""
    db = SessionLocal()
    try:
        reports = db.query(IncidentReport).order_by(IncidentReport.timestamp.desc()).all()
        df = export_incident_reports_to_lab_maintenance_csv(db, reports)
        filename = f"incident_reports_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_path = DATA_DIR / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_path, index=False)

        global latest_report_training_path, training_data_path
        latest_report_training_path = export_path
        training_data_path = export_path

        return jsonify({
            "message": "Exported report training CSV successfully",
            "filename": filename,
            "records": len(df),
            "filepath": str(export_path)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ============================================================================
# API Routes - Maintenance Records
# ============================================================================

@app.route('/api/maintenance', methods=['GET'])
def get_maintenance_records():
    """Get all maintenance records."""
    db = SessionLocal()
    try:
        asset_id = request.args.get('asset_id')
        query = db.query(MaintenanceRecord)
        
        if asset_id:
            query = query.filter(MaintenanceRecord.asset_id == asset_id)
        
        records = query.order_by(MaintenanceRecord.maintenance_date.desc()).all()
        maintenance_records = []

        for record in records:
            record_data = cast(Any, record)
            maintenance_date = cast(Any, record_data.maintenance_date)
            completed_date = cast(Any, record_data.completed_date)

            maintenance_records.append({
                "id": record_data.id,
                "asset_id": record_data.asset_id,
                "maintenance_date": maintenance_date.isoformat() if maintenance_date is not None else None,
                "maintenance_type": record_data.maintenance_type,
                "description": record_data.description,
                "status": record_data.status,
                "technician_name": record_data.technician_name,
                "completed_date": completed_date.isoformat() if completed_date is not None else None,
                "completion_notes": record_data.completion_notes,
                "parts_replaced": record_data.parts_replaced,
                "actions_taken": record_data.actions_taken,
            })
        
        return jsonify({
            "count": len(maintenance_records),
            "records": maintenance_records
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/maintenance', methods=['POST'])
def create_maintenance_record():
    """Create a new maintenance record (schedule maintenance)."""
    db = SessionLocal()
    try:
        data = request.get_json() or {}
        
        # Validate required fields
        required_fields = ['asset_id', 'maintenance_type', 'description']
        for field in required_fields:
            if field not in data or not str(data.get(field, '')).strip():
                return jsonify({"error": f"Missing required field: {field}"}), 400

        asset_id = str(data['asset_id']).strip()
        linked_asset = db.query(Equipment).filter(Equipment.asset_id == asset_id).first()
        if linked_asset is None:
            return jsonify({"error": "Selected asset does not exist"}), 400
        
        # Create new maintenance record
        new_record = MaintenanceRecord(
            asset_id=asset_id,
            maintenance_date=datetime.fromisoformat(data.get('maintenance_date', datetime.now().isoformat())),
            maintenance_type=data['maintenance_type'],
            description=data['description'],
            status=data.get('status', 'scheduled')
        )
        
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        refresh_prediction_cache()
        
        return jsonify({
            "message": "Maintenance record created successfully",
            "record": {
                "id": new_record.id,
                "asset_id": new_record.asset_id,
                "maintenance_date": new_record.maintenance_date.isoformat(),
                "maintenance_type": new_record.maintenance_type,
                "description": new_record.description,
                "status": new_record.status
            }
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route('/api/maintenance/<int:record_id>/complete', methods=['POST'])
def complete_maintenance(record_id):
    """Complete a maintenance record."""
    db = SessionLocal()
    try:
        record = db.query(MaintenanceRecord).filter(MaintenanceRecord.id == record_id).first()
        
        if not record:
            return jsonify({"error": "Maintenance record not found"}), 404
        
        data = request.get_json() or {}
        maintenance_record = cast(Any, record)
        
        # Update maintenance record with completion details
        maintenance_record.status = 'completed'
        maintenance_record.completed_date = datetime.utcnow()
        maintenance_record.technician_name = data.get('technician_name')
        maintenance_record.completion_notes = data.get('completion_notes')
        maintenance_record.parts_replaced = data.get('parts_replaced')
        maintenance_record.actions_taken = data.get('actions_taken')

        linked_equipment = db.query(Equipment).filter(Equipment.asset_id == record.asset_id).first()
        if linked_equipment is not None:
            linked_equipment_record = cast(Any, linked_equipment)
            linked_equipment_record.status = 'active'

        db.commit()
        db.refresh(record)
        refresh_prediction_cache()

        return jsonify({
            "message": "Maintenance completed successfully",
            "record": {
                "id": record.id,
                "asset_id": record.asset_id,
                "status": record.status,
                "completed_date": record.completed_date.isoformat()
            }
        })
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


if __name__ == '__main__':
    app.run(
        debug=os.getenv('DASHBOARD_DEBUG', '0') == '1',
        host='127.0.0.1',
        port=int(os.getenv('PORT', '5000')),
        use_reloader=False,
    )
