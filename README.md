# 🏠 NYC Airbnb Price Prediction

### *Machine Learning Model for Predicting Airbnb Listing Prices in New York City*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)\
![FastAPI](https://img.shields.io/badge/FastAPI-API%20Service-teal.svg)\
![Docker](https://img.shields.io/badge/Docker-Containerized%20App-blue.svg)\
![Fly.io](https://img.shields.io/badge/Deployment-Fly.io-purple.svg)\
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📘 Overview

This repository contains a complete machine learning pipeline to
**predict nightly Airbnb prices in New York City** using listing
metadata such as location, room type, availability, and accommodation
capacity.

The project includes:

-   Data exploration and preprocessing\
-   Model training and comparison\
-   Deployment using FastAPI\
-   Cloud hosting using Fly.io\
-   Fully containerized workflow via Docker

------------------------------------------------------------------------

## 📌 Problem Definition

The objective is to develop a regression model capable of **estimating
nightly Airbnb listing prices** using structured listing information.

The features leveraged include:

-   **Location** (borough → encoded)\
-   **Room type**\
-   **Minimum nights**\
-   **Number of reviews**\
-   **Availability throughout the year**\
-   **Accommodation capacity**

Accurate predictions assist both **hosts** (pricing strategies) and
**guests** (price expectation).

------------------------------------------------------------------------

## 📂 Dataset

This project uses the **New York City Airbnb Open Data** from Kaggle:

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/data

> A local copy (`AB_NYC_2019.csv`) is already included in the
> repository.

------------------------------------------------------------------------

## 🧠 Model Development

The following models were trained and evaluated:

-   **Linear Regression**\
-   **Decision Tree Regressor**\
-   **Random Forest Regressor**\
-   **Gradient Boosting Regressor**\
-   **XGBoost Regressor**

After comparison, the best model was exported as:\
👉 **model.bin**

------------------------------------------------------------------------

## 📊 Model Evaluation

The evaluation metric used is:

### ✔️ RMSE --- Root Mean Squared Error

This measures the average deviation between predicted and actual listing
prices.

------------------------------------------------------------------------

## 🔮 Predictions & Insights

The trained model can be used to predict Airbnb prices in NYC given a
set of listing features.\
The deployed API allows real-time evaluations of listing prices.

------------------------------------------------------------------------

## 🏗️ System Architecture

                +------------------+
                |  Airbnb Dataset  |
                +--------+---------+
                         |
                         v
                +------------------+
                | Data Processing  |
                |  & Feature Eng.  |
                +--------+---------+
                         |
                         v
                +------------------+
                |  Model Training  |
                | (Multiple Models)|
                +--------+---------+
                         |
                         v
                +------------------+
                |  Best Model      |
                |   model.bin      |
                +--------+---------+
                         |
                         v
            +--------------------------------+
            | FastAPI Service (predict.py)   |
            +----------------+---------------+
                             |
                             v
                  +----------------------+
                  |  Fly.io Deployment   |
                  +----------+-----------+
                             |
                             v
                    User → API → Prediction

------------------------------------------------------------------------

## 📁 Repository Structure

  -----------------------------------------------------------------------------
  File                         Description
  ---------------------------- ------------------------------------------------
  **AB_NYC_2019.csv**          Dataset used for training/testing

  **Dockerfile**               Docker image config

  **model.bin**                Pickled final model

  **Mid_Term_Project.ipynb**   Full EDA + model comparison

  **Model_Deploy.ipynb**       Notebook generating model.bin

  **pyproject.toml**,          Environment files for dependency management
  **uv.lock**                  

  **predict.py**               FastAPI prediction service

  **train.py**                 Model training script

  **test.py**                  API testing script

  **requirements.txt**         Project dependencies

  **fly.toml**                 Configuration file generated for the deployed project: little-glade-5122



------------------------------------------------------------------------

## ▶️ Running the Project Locally

### **Prerequisites**

-   Python 3.x\
-   Jupyter Notebook\
-   VS Code (optional)\
-   `uv` dependency manager\
-   FastAPI\
-   Docker (optional for deployment)

------------------------------------------------------------------------

### **Steps**

#### 1. Clone the repository

``` bash
git clone https://github.com/juangrau/DTC-ML-course.git
cd DTC-ML-course
```

#### 2. (Optional) Build & run the Docker container

``` bash
docker build -t predict-house-prices .
docker run -it -p 9696:9696 predict-house-prices:latest
```

#### 3. Test the model locally

``` bash
python test.py
```

#### 4. Open FastAPI UI

    http://localhost:9696/docs#/default/predict_predict_post

------------------------------------------------------------------------

## ☁️ Cloud Deployment (Fly.io)

The model is deployed publicly here:

👉
**https://little-glade-5122.fly.dev/docs#/default/predict_predict_post**

You can send requests directly to this endpoint.

------------------------------------------------------------------------

## 🔑 API Feature Encodings

### **Neighborhood Encoding**

``` json
{
  "manhattan": 1,
  "brooklyn": 0,
  "queens": 2,
  "bronx": 4,
  "staten island": 3,
  "staten_island": 3
}
```

### **Room Type Encoding**

``` json
{
  "entire home/apt": 1,
  "entire_home/apt": 1,
  "private room": 0,
  "private_room": 0,
  "shared room": 2,
  "shared_room": 2
}
```

------------------------------------------------------------------------

## 🧪 Example API Request

### **POST** `/predict`

``` json
{
  "neighbourhood_group": "manhattan",
  "room_type": "entire home/apt",
  "minimum_nights": 3,
  "calculated_host_listings_count": 120,
  "availability_365": 45
}
```

### Example Response

``` json
{
  "predicted_price": 210.56
}
```

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Bayesian hyperparameter optimization\
-   More advanced feature engineering\
-   Interactive dashboard (Streamlit or Dash)\
-   Kubernetes deployment for scalability

------------------------------------------------------------------------

