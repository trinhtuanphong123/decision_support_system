"""
Data Validation & Quality Checks
Kiểm tra chất lượng dữ liệu đầu vào, phát hiện outliers và anomalies
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

class DataValidator:
    """Validate input data quality and detect anomalies"""
    
    def __init__(self):
        """Initialize validator with expected ranges and distributions"""
        
        # Expected ranges for numeric features (based on training data)
        self.feature_ranges = {
            "minimum_nights": {
                "min": 1,
                "max": 365,
                "typical_max": 30,  # Most listings < 30 days
                "warning_threshold": 90
            },
            "calculated_host_listings_count": {
                "min": 0,
                "max": 500,
                "typical_max": 50,
                "warning_threshold": 100
            },
            "availability_365": {
                "min": 0,
                "max": 365,
                "typical_range": (30, 365)
            }
        }
        
        # Expected categorical values
        self.valid_categories = {
            "neighbourhood_group": ["manhattan", "brooklyn", "queens", "bronx", "staten island", "staten_island"],
            "room_type": ["entire home/apt", "entire_home/apt", "private room", "private_room", "shared room", "shared_room"]
        }
        
        # Statistical tracking for anomaly detection
        self.historical_stats = {
            "minimum_nights": {"mean": 7, "std": 10},
            "calculated_host_listings_count": {"mean": 7, "std": 25},
            "availability_365": {"mean": 112, "std": 131}
        }
    
    def validate_input(self, data: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate input data
        
        Args:
            data: Input data dictionary
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = [
            "neighbourhood_group",
            "room_type", 
            "minimum_nights",
            "calculated_host_listings_count",
            "availability_365"
        ]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors, warnings
        
        # Validate neighbourhood_group
        neighbourhood = str(data["neighbourhood_group"]).lower().strip()
        if neighbourhood not in self.valid_categories["neighbourhood_group"]:
            errors.append(
                f"Invalid neighbourhood_group: '{data['neighbourhood_group']}'. "
                f"Valid options: {self.valid_categories['neighbourhood_group']}"
            )
        
        # Validate room_type
        room_type = str(data["room_type"]).lower().strip().replace(" ", "_")
        valid_room_types = [rt.replace(" ", "_") for rt in self.valid_categories["room_type"]]
        if room_type not in valid_room_types:
            errors.append(
                f"Invalid room_type: '{data['room_type']}'. "
                f"Valid options: {self.valid_categories['room_type']}"
            )
        
        # Validate numeric ranges
        for feature, ranges in self.feature_ranges.items():
            if feature not in data:
                continue
            
            value = data[feature]
            
            # Check type
            if not isinstance(value, (int, float)):
                errors.append(f"{feature} must be numeric, got {type(value)}")
                continue
            
            # Check min/max bounds
            if value < ranges["min"] or value > ranges["max"]:
                errors.append(
                    f"{feature} out of valid range [{ranges['min']}, {ranges['max']}]. Got: {value}"
                )
            
            # Check for unusual values (warnings)
            if "typical_max" in ranges and value > ranges["typical_max"]:
                warnings.append(
                    f"{feature} is unusually high ({value}). "
                    f"Typical maximum is {ranges['typical_max']}. Prediction may be less reliable."
                )
            
            if "warning_threshold" in ranges and value > ranges["warning_threshold"]:
                warnings.append(
                    f"{feature} exceeds warning threshold ({value} > {ranges['warning_threshold']})"
                )
        
        # Logical consistency checks
        minimum_nights = data.get("minimum_nights", 0)
        availability = data.get("availability_365", 0)
        
        if minimum_nights > availability:
            warnings.append(
                f"minimum_nights ({minimum_nights}) is greater than availability_365 ({availability}). "
                "This is unusual and may indicate data quality issues."
            )
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    def detect_outliers(self, data: Dict) -> Dict[str, bool]:
        """
        Detect outliers using z-score method
        
        Args:
            data: Input data dictionary
        
        Returns:
            Dictionary mapping feature names to outlier status
        """
        outliers = {}
        
        for feature, stats in self.historical_stats.items():
            if feature not in data:
                continue
            
            value = data[feature]
            mean = stats["mean"]
            std = stats["std"]
            
            # Calculate z-score
            z_score = abs((value - mean) / std) if std > 0 else 0
            
            # Flag as outlier if z-score > 3
            is_outlier = z_score > 3
            outliers[feature] = is_outlier
            
            if is_outlier:
                outliers[f"{feature}_z_score"] = round(z_score, 2)
        
        return outliers
    
    def get_data_quality_score(self, data: Dict) -> Dict:
        """
        Calculate overall data quality score
        
        Returns:
            Dictionary with quality score and breakdown
        """
        is_valid, errors, warnings = self.validate_input(data)
        outliers = self.detect_outliers(data)
        
        # Calculate score (0-100)
        score = 100
        
        # Deduct for errors
        score -= len(errors) * 25
        
        # Deduct for warnings
        score -= len(warnings) * 10
        
        # Deduct for outliers
        num_outliers = sum(1 for k, v in outliers.items() if not k.endswith("_z_score") and v)
        score -= num_outliers * 15
        
        score = max(0, score)
        
        # Determine quality level
        if score >= 90:
            quality_level = "excellent"
        elif score >= 75:
            quality_level = "good"
        elif score >= 50:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "quality_score": score,
            "quality_level": quality_level,
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "outliers": outliers,
            "recommendations": self._get_recommendations(errors, warnings, outliers)
        }
    
    def _get_recommendations(self, errors: List[str], warnings: List[str], outliers: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if errors:
            recommendations.append("Fix data errors before making prediction")
        
        if warnings:
            recommendations.append("Review warnings - prediction may be less reliable")
        
        num_outliers = sum(1 for k, v in outliers.items() if not k.endswith("_z_score") and v)
        if num_outliers > 0:
            recommendations.append(
                f"Detected {num_outliers} outlier(s). Consider if values are realistic."
            )
        
        if not recommendations:
            recommendations.append("Data quality looks good!")
        
        return recommendations


# ==========================================
# FastAPI Integration
# ==========================================

class ValidationResult:
    """Validation result with detailed feedback"""
    
    def __init__(self, 
                 is_valid: bool,
                 quality_score: int,
                 quality_level: str,
                 errors: List[str],
                 warnings: List[str],
                 outliers: Dict,
                 recommendations: List[str]):
        self.is_valid = is_valid
        self.quality_score = quality_score
        self.quality_level = quality_level
        self.errors = errors
        self.warnings = warnings
        self.outliers = outliers
        self.recommendations = recommendations
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level,
            "errors": self.errors,
            "warnings": self.warnings,
            "outliers": self.outliers,
            "recommendations": self.recommendations
        }


# Global validator instance
data_validator = DataValidator()


def get_validator() -> DataValidator:
    """Get global validator instance"""
    return data_validator


# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    validator = DataValidator()
    
    # Test case 1: Valid data
    good_data = {
        "neighbourhood_group": "manhattan",
        "room_type": "entire home/apt",
        "minimum_nights": 3,
        "calculated_host_listings_count": 5,
        "availability_365": 200
    }
    
    result = validator.get_data_quality_score(good_data)
    print("Good data quality:", result)
    print()
    
    # Test case 2: Data with warnings
    warning_data = {
        "neighbourhood_group": "brooklyn",
        "room_type": "private room",
        "minimum_nights": 45,  # Unusually high
        "calculated_host_listings_count": 150,  # Unusually high
        "availability_365": 10  # Very low
    }
    
    result = validator.get_data_quality_score(warning_data)
    print("Warning data quality:", result)
    print()
    
    # Test case 3: Invalid data
    bad_data = {
        "neighbourhood_group": "invalid_borough",
        "room_type": "invalid_type",
        "minimum_nights": -5,  # Invalid
        "calculated_host_listings_count": 1000,  # Out of range
        "availability_365": 400  # Out of range
    }
    
    result = validator.get_data_quality_score(bad_data)
    print("Bad data quality:", result)