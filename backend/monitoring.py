"""
Model Monitoring & Metrics Tracking
Thu thập metrics về model performance, prediction distribution, và system health
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import json
import os



class ModelMonitor:
    """Track model predictions and performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize monitoring system
        
        Args:
            max_history: Maximum number of predictions to keep in memory
        """
        self.max_history = max_history
        
        # Prediction history (in-memory queue)
        self.prediction_history = deque(maxlen=max_history)
        
        # Metrics counters
        self.metrics = {
            "total_predictions": 0,
            "predictions_by_confidence": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "predictions_by_neighbourhood": {},
            "predictions_by_room_type": {},
            "average_prediction_time_ms": 0,
            "error_count": 0,
            "min_price_predicted": float('inf'),
            "max_price_predicted": 0,
            "sum_predictions": 0.0
        }
        
        # Performance tracking
        self.prediction_times = deque(maxlen=100)  # Last 100 prediction times
        
        # Data drift detection (simplified)
        self.feature_distributions = {
            "minimum_nights": deque(maxlen=1000),
            "availability_365": deque(maxlen=1000),
            "host_listings_count": deque(maxlen=1000)
        }
    
    def log_prediction(self, 
                      input_features: Dict,
                      prediction: float,
                      confidence: str,
                      processing_time_ms: float,
                      timestamp: Optional[datetime] = None):
        """
        Log a prediction event
        
        Args:
            input_features: Input features used for prediction
            prediction: Predicted price
            confidence: Confidence level (high/medium/low)
            processing_time_ms: Time taken to process prediction
            timestamp: Timestamp of prediction (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create prediction record
        record = {
            "timestamp": timestamp.isoformat(),
            "input": input_features,
            "prediction": prediction,
            "confidence": confidence,
            "processing_time_ms": processing_time_ms
        }
        
        # Add to history
        self.prediction_history.append(record)
        
        # Update metrics
        self.metrics["total_predictions"] += 1
        self.metrics["predictions_by_confidence"][confidence] += 1
        
        # Track by neighbourhood
        neighbourhood = input_features.get("neighbourhood_group", "unknown")
        self.metrics["predictions_by_neighbourhood"][neighbourhood] = \
            self.metrics["predictions_by_neighbourhood"].get(neighbourhood, 0) + 1
        
        # Track by room type
        room_type = input_features.get("room_type", "unknown")
        self.metrics["predictions_by_room_type"][room_type] = \
            self.metrics["predictions_by_room_type"].get(room_type, 0) + 1
        
        # Update price statistics
        self.metrics["min_price_predicted"] = min(
            self.metrics["min_price_predicted"], 
            prediction
        )
        self.metrics["max_price_predicted"] = max(
            self.metrics["max_price_predicted"], 
            prediction
        )
        self.metrics["sum_predictions"] += prediction
        
        # Track processing time
        self.prediction_times.append(processing_time_ms)
        self.metrics["average_prediction_time_ms"] = sum(self.prediction_times) / len(self.prediction_times)
        
        # Track feature distributions for drift detection
        self.feature_distributions["minimum_nights"].append(
            input_features.get("minimum_nights", 0)
        )
        self.feature_distributions["availability_365"].append(
            input_features.get("availability_365", 0)
        )
        self.feature_distributions["host_listings_count"].append(
            input_features.get("calculated_host_listings_count", 0)
        )
    
    def log_error(self, error_type: str, error_message: str):
        """Log an error event"""
        self.metrics["error_count"] += 1
        
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message
        }
        
        # Could save to file or database here
        print(f"Error logged: {error_record}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics summary"""
        total = self.metrics["total_predictions"]
        
        metrics_summary = {
            "total_predictions": total,
            "error_count": self.metrics["error_count"],
            "error_rate": self.metrics["error_count"] / max(total, 1),
            
            "confidence_distribution": self.metrics["predictions_by_confidence"],
            
            "predictions_by_neighbourhood": self.metrics["predictions_by_neighbourhood"],
            "predictions_by_room_type": self.metrics["predictions_by_room_type"],
            
            "price_statistics": {
                "min": self.metrics["min_price_predicted"] if total > 0 else 0,
                "max": self.metrics["max_price_predicted"],
                "average": self.metrics["sum_predictions"] / max(total, 1)
            },
            
            "performance": {
                "average_prediction_time_ms": round(self.metrics["average_prediction_time_ms"], 2),
                "recent_predictions": len(self.prediction_history)
            }
        }
        
        return metrics_summary
    
    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Get n most recent predictions"""
        return list(self.prediction_history)[-n:]
    
    def detect_data_drift(self) -> Dict:
        """
        Simple data drift detection
        Compare recent data distribution vs initial baseline
        """
        if len(self.feature_distributions["minimum_nights"]) < 100:
            return {"status": "insufficient_data"}
        
        # Compare first 100 vs last 100 predictions
        drift_report = {}
        
        for feature, values in self.feature_distributions.items():
            values_list = list(values)
            
            # Baseline: first 100
            baseline = values_list[:100]
            baseline_avg = sum(baseline) / len(baseline)
            
            # Recent: last 100
            recent = values_list[-100:]
            recent_avg = sum(recent) / len(recent)
            
            # Calculate drift (% change)
            drift_pct = ((recent_avg - baseline_avg) / max(baseline_avg, 1)) * 100
            
            drift_report[feature] = {
                "baseline_avg": round(baseline_avg, 2),
                "recent_avg": round(recent_avg, 2),
                "drift_percentage": round(drift_pct, 2),
                "status": "drifted" if abs(drift_pct) > 20 else "stable"
            }
        
        return drift_report
    
    def export_metrics(self, filepath: str = "metrics.json"):
        """Export metrics to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self.get_metrics(),
                    "drift_detection": self.detect_data_drift(),
                    "recent_predictions": self.get_recent_predictions(50)
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to export metrics: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict:
        """Get data formatted for monitoring dashboard"""
        metrics = self.get_metrics()
        drift = self.detect_data_drift()
        
        return {
            "overview": {
                "total_predictions": metrics["total_predictions"],
                "error_rate": f"{metrics['error_rate']*100:.2f}%",
                "avg_response_time": f"{metrics['performance']['average_prediction_time_ms']:.2f}ms",
                "avg_predicted_price": f"${metrics['price_statistics']['average']:.2f}"
            },
            "confidence_breakdown": metrics["confidence_distribution"],
            "neighbourhood_breakdown": metrics["predictions_by_neighbourhood"],
            "room_type_breakdown": metrics["predictions_by_room_type"],
            "price_range": {
                "min": metrics["price_statistics"]["min"],
                "max": metrics["price_statistics"]["max"],
                "avg": metrics["price_statistics"]["average"]
            },
            "drift_detection": drift,
            "recent_activity": self.get_recent_predictions(20)
        }


# Global monitor instance
model_monitor = ModelMonitor(max_history=1000)


# ==========================================
# Integration with FastAPI
# ==========================================

def get_monitor() -> ModelMonitor:
    """Get the global monitor instance"""
    return model_monitor