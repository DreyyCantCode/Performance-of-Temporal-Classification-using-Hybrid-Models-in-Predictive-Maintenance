"""Database setup and models for MySQL integration."""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'lab_monitoring')
USE_SQLITE = os.getenv('USE_SQLITE', 'True').lower() == 'true'  # Default to SQLite for easier setup

# Connection string
if USE_SQLITE:
    DATABASE_URL = f"sqlite:///./lab_monitoring.db"
else:
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for models
Base = declarative_base()


class Equipment(Base):
    """Equipment/Asset model."""
    __tablename__ = "equipment"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(50), unique=True, index=True)
    name = Column(String(255))
    description = Column(String(500))
    location = Column(String(255))
    shift_type = Column(String(20), default="am")  # am, pm
    uptime_hours = Column(Float, default=10.0)  # Derived from shift schedule
    status = Column(String(20), default="active")  # active, inactive, maintenance
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SensorReading(Base):
    """Sensor/Metric readings from equipment."""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    metric_1 = Column(Float)  # Equipment Health
    metric_2 = Column(Float)  # Resource Utilization
    metric_3 = Column(Float)  # System Stability
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """ML model predictions."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    sensor_reading_id = Column(Integer)
    model_type = Column(String(50))  # xgboost, bilstm, etc
    failure_risk = Column(Float)  # 0-1 probability
    will_fail_24h = Column(Boolean)  # True if risk > threshold
    confidence = Column(Float)  # Model confidence
    created_at = Column(DateTime, default=datetime.utcnow)


class MaintenanceRecord(Base):
    """Maintenance history."""
    __tablename__ = "maintenance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(50), index=True)
    maintenance_date = Column(DateTime, index=True)
    maintenance_type = Column(String(100))  # preventive, corrective, emergency
    description = Column(String(1000))
    status = Column(String(50))  # scheduled, in_progress, completed
    
    # Technician/completion details
    technician_name = Column(String(255), nullable=True)
    completed_date = Column(DateTime, nullable=True)
    completion_notes = Column(String(2000), nullable=True)
    
    # Parts/actions taken
    parts_replaced = Column(String(500), nullable=True)
    actions_taken = Column(String(1000), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class IncidentReport(Base):
    """Incident reports submitted by users."""
    __tablename__ = "incident_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(50), index=True, nullable=True)
    component = Column(String(100))  # PC, Projector, Router, etc
    room = Column(String(100))  # Lab 1, Lab 2, etc
    reported_by = Column(String(255))
    date = Column(String(20))  # Date string from form
    priority = Column(String(20))  # Low, Medium, High, Critical
    affected_users = Column(Integer, nullable=True)
    description = Column(String(2000))
    status = Column(String(50), default="In Progress")  # In Progress, Completed
    completed_on = Column(String(20), nullable=True)  # Completion date
    health_score = Column(Float, nullable=True)
    performance_score = Column(Float, nullable=True)
    timestamp = Column(Float)  # Unix timestamp for sorting
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    ensure_equipment_schedule_columns()
    ensure_maintenance_record_columns()
    ensure_incident_report_columns()
    print("✓ Database tables created successfully")


def ensure_equipment_schedule_columns():
    """Add uptime-related equipment columns if they are missing."""
    inspector = inspect(engine)
    if 'equipment' not in inspector.get_table_names():
        return

    existing_columns = {column['name'] for column in inspector.get_columns('equipment')}
    alter_statements = []

    if 'shift_type' not in existing_columns:
        alter_statements.append("ALTER TABLE equipment ADD COLUMN shift_type VARCHAR(20) DEFAULT 'am'")
    if 'uptime_hours' not in existing_columns:
        alter_statements.append("ALTER TABLE equipment ADD COLUMN uptime_hours FLOAT DEFAULT 10.0")

    if alter_statements:
        with engine.begin() as connection:
            for statement in alter_statements:
                connection.execute(text(statement))


def ensure_incident_report_columns():
    """Add report linkage columns if they are missing."""
    inspector = inspect(engine)
    if 'incident_reports' not in inspector.get_table_names():
        return

    existing_columns = {column['name'] for column in inspector.get_columns('incident_reports')}
    alter_statements = []

    if 'asset_id' not in existing_columns:
        alter_statements.append("ALTER TABLE incident_reports ADD COLUMN asset_id VARCHAR(50)")
    if 'affected_users' not in existing_columns:
        alter_statements.append("ALTER TABLE incident_reports ADD COLUMN affected_users INTEGER")
    if 'health_score' not in existing_columns:
        alter_statements.append("ALTER TABLE incident_reports ADD COLUMN health_score FLOAT")
    if 'performance_score' not in existing_columns:
        alter_statements.append("ALTER TABLE incident_reports ADD COLUMN performance_score FLOAT")

    if alter_statements:
        with engine.begin() as connection:
            for statement in alter_statements:
                connection.execute(text(statement))


def ensure_maintenance_record_columns():
    """Add maintenance history columns if they are missing."""
    inspector = inspect(engine)
    if 'maintenance_records' not in inspector.get_table_names():
        return

    existing_columns = {column['name'] for column in inspector.get_columns('maintenance_records')}
    alter_statements = []

    if 'technician_name' not in existing_columns:
        alter_statements.append("ALTER TABLE maintenance_records ADD COLUMN technician_name VARCHAR(255)")
    if 'completed_date' not in existing_columns:
        alter_statements.append("ALTER TABLE maintenance_records ADD COLUMN completed_date DATETIME")
    if 'completion_notes' not in existing_columns:
        alter_statements.append("ALTER TABLE maintenance_records ADD COLUMN completion_notes VARCHAR(2000)")
    if 'parts_replaced' not in existing_columns:
        alter_statements.append("ALTER TABLE maintenance_records ADD COLUMN parts_replaced VARCHAR(500)")
    if 'actions_taken' not in existing_columns:
        alter_statements.append("ALTER TABLE maintenance_records ADD COLUMN actions_taken VARCHAR(1000)")

    if alter_statements:
        with engine.begin() as connection:
            for statement in alter_statements:
                connection.execute(text(statement))


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == '__main__':
    # Test connection and create tables
    try:
        create_tables()
        print("✓ Connected to MySQL database successfully")
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        print("\nMake sure MySQL is running and configured in .env file:")
        print("  DB_USER=root")
        print("  DB_PASSWORD=your_password")
        print("  DB_HOST=localhost")
        print("  DB_PORT=3306")
        print("  DB_NAME=lab_monitoring")
