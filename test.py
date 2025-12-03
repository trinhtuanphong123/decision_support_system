import requests

#url = 'http://localhost:9696/predict'

url = "https://little-glade-5122.fly.dev/predict"


listing = {
  "neighbourhood_group": "brooklyn",
  "room_type": 0,
  "minimum_nights": 0,
  "calculated_host_listings_count": 0,
  "availability_365": 0
}

response = requests.post(url, json=listing)

predictions = response.json()

print(f"Predicted price for the listing: {predictions}")
