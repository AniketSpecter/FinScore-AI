from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # allow anonymous predictions
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Store raw input features as JSON
    input_data = Column(JSON)
    
    # Store outputs
    risk_score = Column(Float)
    probability_default = Column(Float)
    recommendation = Column(String)
    risk_category = Column(String)
    
    # Model used
    model_version = Column(String)
    
    user = relationship("User", back_populates="predictions")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String, unique=True, index=True)
    deployed_at = Column(DateTime, default=datetime.datetime.utcnow)
    model_path = Column(String)
    metrics = Column(JSON) # e.g. f1 score, accuracy
    is_active = Column(Integer, default=1) # 1 for active, 0 for inactive

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    action = Column(String) # e.g., "model_retrained", "prediction_made", "service_started"
    details = Column(String)
