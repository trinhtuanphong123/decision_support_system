import requests
import json

# Configuration
BASE_URL = "https://airbnb-price-predictor-v41y.onrender.com"  # Change to your deployed URL later

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*50)
    print("Testing Health Check Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_encodings():
    """Test encodings endpoint"""
    print("\n" + "="*50)
    print("Testing Encodings Endpoint")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/encodings")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction_with_text():
    """Test prediction with text input"""
    print("\n" + "="*50)
    print("Test 1: Prediction with Text Input")
    print("="*50)
    
    listing = {
        "neighbourhood_group": "manhattan",
        "room_type": "entire home/apt",
        "minimum_nights": 3,
        "calculated_host_listings_count": 5,
        "availability_365": 200
    }
    
    print(f"Input: {json.dumps(listing, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=listing)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Price: ${result['price_prediction']:.2f}")
    else:
        print(f"Error: {response.text}")

def test_prediction_with_numbers():
    """Test prediction with numeric input"""
    print("\n" + "="*50)
    print("Test 2: Prediction with Numeric Input")
    print("="*50)
    
    listing = {
        "neighbourhood_group": 0,  # Brooklyn
        "room_type": 0,            # Private room
        "minimum_nights": 2,
        "calculated_host_listings_count": 10,
        "availability_365": 365
    }
    
    print(f"Input: {json.dumps(listing, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=listing)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Price: ${result['price_prediction']:.2f}")
    else:
        print(f"Error: {response.text}")

def test_multiple_scenarios():
    """Test multiple scenarios"""
    print("\n" + "="*50)
    print("Test 3: Multiple Scenarios")
    print("="*50)
    
    scenarios = [
        {
            "name": "Luxury Manhattan Entire Home",
            "data": {
                "neighbourhood_group": "manhattan",
                "room_type": "entire home/apt",
                "minimum_nights": 7,
                "calculated_host_listings_count": 1,
                "availability_365": 90
            }
        },
        {
            "name": "Budget Brooklyn Private Room",
            "data": {
                "neighbourhood_group": "brooklyn",
                "room_type": "private room",
                "minimum_nights": 1,
                "calculated_host_listings_count": 3,
                "availability_365": 365
            }
        },
        {
            "name": "Queens Shared Room",
            "data": {
                "neighbourhood_group": "queens",
                "room_type": "shared room",
                "minimum_nights": 1,
                "calculated_host_listings_count": 2,
                "availability_365": 180
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        response = requests.post(f"{BASE_URL}/predict", json=scenario['data'])
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Price: ${result['price_prediction']:.2f}")
        else:
            print(f"Error: {response.text}")

def test_invalid_input():
    """Test with invalid input"""
    print("\n" + "="*50)
    print("Test 4: Invalid Input Handling")
    print("="*50)
    
    invalid_listing = {
        "neighbourhood_group": "invalid_place",
        "room_type": "entire home/apt",
        "minimum_nights": 3,
        "calculated_host_listings_count": 5,
        "availability_365": 200
    }
    
    print(f"Input: {json.dumps(invalid_listing, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_listing)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("NYC AIRBNB PRICE PREDICTION API - TEST SUITE")
    print("="*50)
    print(f"Testing API at: {BASE_URL}")
    
    try:
        # Test health first
        if not test_health():
            print("\n❌ Health check failed! Make sure the API is running.")
            return
        
        # Get valid encodings
        test_encodings()
        
        # Run prediction tests
        test_prediction_with_text()
        test_prediction_with_numbers()
        test_multiple_scenarios()
        test_invalid_input()
        
        print("\n" + "="*50)
        print("✅ All tests completed!")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error!")
        print(f"Could not connect to {BASE_URL}")
        print("Make sure the API is running:")
        print("  - Local: python predict.py")
        print("  - Docker: docker run -p 9696:9696 airbnb-predictor")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()