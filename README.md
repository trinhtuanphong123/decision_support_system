# 🏠 NYC Airbnb Price Prediction API

> Machine Learning API for predicting Airbnb listing prices in New York City

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121.2-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live Demo**: [https://airbnb-price-predictor-v41y.onrender.com/docs](https://airbnb-price-predictor-v41y.onrender.com/docs)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project provides a REST API for predicting nightly Airbnb listing prices in New York City using machine learning. The model is trained on the [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) dataset.

### Problem Statement

Predict the nightly price of Airbnb listings based on:
- Location (neighbourhood)
- Room type
- Minimum nights required
- Host's total listings
- Availability throughout the year

---

## ✨ Features

- 🤖 **XGBoost ML Model** with RMSE of 0.482
- ⚡ **Fast API** with automatic OpenAPI documentation
- 🐳 **Docker Support** for easy deployment
- 🔒 **Input Validation** using Pydantic models
- 📊 **Health Checks** and monitoring endpoints
- 🌐 **CORS Enabled** for web integration
- 🎨 **Swagger UI** for interactive API testing

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **ML Framework** | XGBoost, scikit-learn |
| **API Framework** | FastAPI, Uvicorn |
| **Language** | Python 3.11 |
| **Containerization** | Docker |
| **Deployment** | Render.com |
| **Data Processing** | NumPy, Pandas |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional but recommended)
- Git

### Option 1: Local Setup (without Docker)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nyc-airbnb-prediction.git
cd nyc-airbnb-prediction

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python predict.py

# API will be available at http://localhost:9696
```

### Option 2: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nyc-airbnb-prediction.git
cd nyc-airbnb-prediction

# Build Docker image
docker build -t airbnb-predictor .

# Run container
docker run -d -p 9696:9696 --name airbnb-api airbnb-predictor

# Check logs
docker logs -f airbnb-api

# API will be available at http://localhost:9696
```

### Test the API

```bash
# Health check
curl http://localhost:9696/health

# Make a prediction
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "neighbourhood_group": "manhattan",
    "room_type": "entire home/apt",
    "minimum_nights": 3,
    "calculated_host_listings_count": 5,
    "availability_365": 200
  }'
```

---

## 📚 API Documentation

### Base URL
- **Local**: `http://localhost:9696`
- **Production**: `https://airbnb-price-predictor-v41y.onrender.com`

### Endpoints

#### 1. Root
```http
GET /
```

**Response:**
```json
{
  "message": "NYC Airbnb Price Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict (POST)",
    "docs": "/docs"
  }
}
```

#### 2. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 3. Predict Price
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "neighbourhood_group": "manhattan",
  "room_type": "entire home/apt",
  "minimum_nights": 3,
  "calculated_host_listings_count": 5,
  "availability_365": 200
}
```

**Input Fields:**

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `neighbourhood_group` | string/int | manhattan, brooklyn, queens, bronx, staten island | NYC borough |
| `room_type` | string/int | entire home/apt, private room, shared room | Type of accommodation |
| `minimum_nights` | int | 0-365 | Minimum nights required |
| `calculated_host_listings_count` | int | 0-N | Total listings by host |
| `availability_365` | int | 0-365 | Days available per year |

**Response:**
```json
{
  "price_prediction": 157.42
}
```

#### 4. Get Valid Encodings
```http
GET /encodings
```

**Response:**
```json
{
  "neighbourhood_group": ["manhattan", "brooklyn", "queens", "bronx", "staten island"],
  "room_type": ["entire home/apt", "private room", "shared room"]
}
```

#### 5. Interactive Documentation
```http
GET /docs
```
Swagger UI for interactive API testing

---

## 📊 Model Performance

### Training Details

- **Algorithm**: XGBoost Regressor
- **Dataset**: NYC Airbnb Open Data 2019 (48,895 listings)
- **Train/Test Split**: 80/20
- **Target**: `log1p(price)` (log-transformed for better distribution)

### Hyperparameters

```python
{
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 10,
    'objective': 'reg:squarederror',
    'num_boost_round': 100
}
```

### Metrics

| Metric | Value |
|--------|-------|
| **RMSE** | 0.482 (on log scale) |
| **Features** | 5 input features after encoding |
| **Model Size** | ~2MB (pickled) |

### Feature Importance

1. `neighbourhood_group` - Location impact
2. `room_type` - Accommodation type
3. `availability_365` - Supply indicator
4. `calculated_host_listings_count` - Host experience
5. `minimum_nights` - Booking constraint

---

## 📁 Project Structure

```
nyc-airbnb-prediction/
│
├── model_training/              # Training notebooks (not deployed)
│   ├── AB_NYC_2019.csv         # Dataset
│   └── Model_Deploy.ipynb      # Model training notebook
│
├── model.bin                    # Trained model (pickle)
├── predict.py                   # FastAPI application
├── test.py                      # API testing script
│
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── .dockerignore               # Docker ignore rules
├── .gitignore                  # Git ignore rules
├── .python-version             # Python version (3.11)
│
└── README.md                   # This file
```

---

## 💻 Development

### Setup Development Environment

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/nyc-airbnb-prediction.git
cd nyc-airbnb-prediction

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with auto-reload (for development)
uvicorn predict:app --reload --host 0.0.0.0 --port 9696
```

### Run Tests

```bash
# Make sure API is running first
python test.py
```

### Retrain Model

```bash
# Navigate to training folder
cd model_training

# Run training notebook or script
jupyter notebook Model_Deploy.ipynb
# or
python train.py

# Copy new model.bin to root directory
cp model.bin ../
```

---

## 🌐 Deployment

### Deploy to Render

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

2. **Create Web Service on Render**
- Go to [Render.com](https://render.com)
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Configure:
  - **Name**: `airbnb-price-predictor`
  - **Runtime**: Docker
  - **Instance Type**: Free
- Click "Create Web Service"

3. **Wait for Deployment**
- Build takes ~5-10 minutes
- Your API will be live at: `https://your-app.onrender.com`

### Deploy to Other Platforms

<details>
<summary>Google Cloud Run</summary>

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/airbnb-predictor

# Deploy
gcloud run deploy airbnb-predictor \
  --image gcr.io/PROJECT_ID/airbnb-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
</details>

<details>
<summary>AWS Elastic Beanstalk</summary>

```bash
# Initialize EB
eb init -p docker airbnb-predictor

# Create environment
eb create airbnb-predictor-env

# Deploy
eb deploy
```
</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

---

## 📝 Future Improvements

- [ ] Add authentication (API keys)
- [ ] Implement rate limiting
- [ ] Add caching layer (Redis)
- [ ] Store prediction history in database
- [ ] Add model versioning and A/B testing
- [ ] Implement CI/CD pipeline (GitHub Actions)
- [ ] Add monitoring (Prometheus + Grafana)
- [ ] Create frontend dashboard
- [ ] Support batch predictions
- [ ] Add more features (reviews, amenities)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your Name](https://linkedin.com/in/YOUR_PROFILE)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- Inspired by: [DataTalks.Club ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- Deployment: [Render.com](https://render.com)

---

## 📞 Support

If you have any questions or issues, please:
1. Check the [FAQ](#faq) section
2. Open an [Issue](https://github.com/YOUR_USERNAME/nyc-airbnb-prediction/issues)
3. Contact me via [email](mailto:your.email@example.com)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by [Your Name]

</div>