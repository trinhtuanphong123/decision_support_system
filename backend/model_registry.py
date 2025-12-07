"""
Model Registry & Version Management
Quản lý nhiều phiên bản model, metadata, và A/B testing
"""

import pickle
import json
import os
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

class ModelVersion:
    """Represents a single model version"""
    
    def __init__(self, 
                 version: str,
                 model_path: str,
                 metadata: Dict):
        self.version = version
        self.model_path = model_path
        self.metadata = metadata
        self.dv = None
        self.model = None
        self.loaded = False
    
    def load(self):
        """Load model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                self.dv, self.model = pickle.load(f)
            self.loaded = True
            return True
        except Exception as e:
            print(f"Failed to load model {self.version}: {e}")
            return False
    
    def predict(self, X):
        """Make prediction"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        return self.model.predict(X)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "loaded": self.loaded
        }


class ModelRegistry:
    """Registry for managing multiple model versions"""
    
    def __init__(self, registry_dir: str = "./models"):
        """
        Initialize model registry
        
        Args:
            registry_dir: Directory containing model files
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.models: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        
        # Load registry metadata
        self.metadata_file = self.registry_dir / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from metadata file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.active_version = data.get("active_version")
                
                # Load model versions
                for version_info in data.get("versions", []):
                    version = ModelVersion(
                        version=version_info["version"],
                        model_path=version_info["model_path"],
                        metadata=version_info["metadata"]
                    )
                    self.models[version.version] = version
    
    def _save_registry(self):
        """Save registry to metadata file"""
        data = {
            "active_version": self.active_version,
            "versions": [m.to_dict() for m in self.models.values()],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self,
                      version: str,
                      model_path: str,
                      metadata: Optional[Dict] = None) -> bool:
        """
        Register a new model version
        
        Args:
            version: Version identifier (e.g., "v1.0", "v2.0")
            model_path: Path to model file
            metadata: Additional metadata (metrics, training date, etc.)
        
        Returns:
            True if successful
        """
        if metadata is None:
            metadata = {}
        
        # Add registration timestamp
        metadata["registered_at"] = datetime.now().isoformat()
        
        # Create model version
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            metadata=metadata
        )
        
        # Store in registry
        self.models[version] = model_version
        
        # Set as active if first model
        if self.active_version is None:
            self.active_version = version
        
        # Save registry
        self._save_registry()
        
        return True
    
    def get_model(self, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific model version
        
        Args:
            version: Version to get (None = active version)
        
        Returns:
            ModelVersion or None if not found
        """
        if version is None:
            version = self.active_version
        
        return self.models.get(version)
    
    def set_active_version(self, version: str) -> bool:
        """
        Set active model version
        
        Args:
            version: Version to activate
        
        Returns:
            True if successful
        """
        if version not in self.models:
            return False
        
        self.active_version = version
        self._save_registry()
        return True
    
    def list_versions(self) -> List[Dict]:
        """List all registered versions"""
        return [
            {
                **m.to_dict(),
                "is_active": m.version == self.active_version
            }
            for m in self.models.values()
        ]
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """
        Compare metadata between two versions
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Comparison dict
        """
        m1 = self.models.get(version1)
        m2 = self.models.get(version2)
        
        if not m1 or not m2:
            return {"error": "Version not found"}
        
        return {
            "version1": {
                "version": m1.version,
                "metadata": m1.metadata
            },
            "version2": {
                "version": m2.version,
                "metadata": m2.metadata
            },
            "comparison": {
                "registered_at_diff": m1.metadata.get("registered_at") != m2.metadata.get("registered_at"),
                "metrics_diff": m1.metadata.get("metrics") != m2.metadata.get("metrics")
            }
        }
    
    def rollback(self) -> Optional[str]:
        """
        Rollback to previous version
        
        Returns:
            Previous version name or None
        """
        versions = sorted(
            self.models.keys(),
            key=lambda v: self.models[v].metadata.get("registered_at", ""),
            reverse=True
        )
        
        if len(versions) < 2:
            return None
        
        # Find current version index
        try:
            current_idx = versions.index(self.active_version)
            if current_idx < len(versions) - 1:
                previous_version = versions[current_idx + 1]
                self.set_active_version(previous_version)
                return previous_version
        except ValueError:
            pass
        
        return None


# ==========================================
# Example Usage
# ==========================================

def create_example_registry():
    """Example: How to use model registry"""
    
    # Initialize registry
    registry = ModelRegistry(registry_dir="./models")
    
    # Register model v1.0
    registry.register_model(
        version="v1.0",
        model_path="./model.bin",
        metadata={
            "algorithm": "XGBoost",
            "training_date": "2024-01-15",
            "metrics": {
                "rmse": 0.25,
                "mae": 0.18,
                "r2": 0.87
            },
            "features": ["neighbourhood_group", "room_type", "minimum_nights", 
                        "calculated_host_listings_count", "availability_365"],
            "training_samples": 49000
        }
    )
    
    # Register model v2.0 (improved version)
    registry.register_model(
        version="v2.0",
        model_path="./model_v2.bin",
        metadata={
            "algorithm": "XGBoost",
            "training_date": "2024-06-20",
            "metrics": {
                "rmse": 0.22,  # Improved
                "mae": 0.16,   # Improved
                "r2": 0.89     # Improved
            },
            "features": ["neighbourhood_group", "room_type", "minimum_nights", 
                        "calculated_host_listings_count", "availability_365"],
            "training_samples": 55000,
            "improvements": "Added more recent data, tuned hyperparameters"
        }
    )
    
    # List all versions
    print("All versions:", registry.list_versions())
    
    # Set active version
    registry.set_active_version("v2.0")
    
    # Get active model
    active_model = registry.get_model()
    print(f"Active version: {active_model.version}")
    
    # Compare versions
    comparison = registry.compare_versions("v1.0", "v2.0")
    print("Comparison:", comparison)
    
    # Rollback if needed
    previous = registry.rollback()
    print(f"Rolled back to: {previous}")


if __name__ == "__main__":
    create_example_registry()