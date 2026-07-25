# Customer Churn Prediction

An end-to-end machine learning application that predicts whether a bank customer is likely to churn. The project covers the complete ML lifecycle — from data preprocessing and model comparison to hyperparameter tuning, REST API development, Docker containerization, and cloud deployment.

## 🚀 Live Demo

**Swagger API Documentation:**
https://customer-churn-prediction-d4i0.onrender.com/docs

The API provides a `/predict` endpoint for real-time customer churn prediction.

> Note: The Render free instance may take some time to wake up after inactivity.

---

## 📌 Problem Statement

Customer churn is an important business problem for banks. Identifying customers who are likely to leave can help organizations take proactive retention measures.

This project uses customer demographic, financial, and account-related information to predict whether a customer will churn.

### Target

* `0` → Customer will not churn
* `1` → Customer will churn

---

## 🎯 Project Objectives

* Perform data preprocessing and feature engineering
* Handle numerical and categorical features using a preprocessing pipeline
* Compare multiple classification algorithms
* Evaluate models using multiple performance metrics
* Perform cross-validation
* Tune the Random Forest model using GridSearchCV
* Save the best trained model
* Build a REST API using FastAPI
* Containerize the application using Docker
* Deploy the API to Render

---

## 🛠️ Tech Stack

### Programming

* Python
* SQL

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Machine Learning Algorithms

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* XGBoost

### Backend & Deployment

* FastAPI
* Uvicorn
* Docker
* Render

### Development Tools

* Git
* GitHub
* VS Code
* Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Cleaning
     ↓
Remove Unnecessary Columns
     ↓
Train-Test Split
     ↓
Numerical & Categorical Feature Identification
     ↓
Preprocessing Pipeline
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Cross Validation
     ↓
Hyperparameter Tuning
     ↓
Best Model Selection
     ↓
Model Serialization
     ↓
FastAPI REST API
     ↓
Docker Container
     ↓
Render Deployment
```

---

## 📊 Dataset

The project uses the **Churn Modelling** dataset containing **10,000 customer records** and **14 original columns**.

After removing unnecessary identifiers, the model uses **10 features**.

### Features

#### Numerical Features

* CreditScore
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

#### Categorical Features

* Geography
* Gender

### Target

* Exited

---

## 🤖 Models Compared

Six classification algorithms were evaluated:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   80.80% |    58.91% | 18.67% |   28.36% |  77.48% |
| Decision Tree       |   78.25% |    46.85% | 51.11% |   48.88% |  68.15% |
| Random Forest       |   86.40% |    78.24% | 45.95% |   57.89% |  85.21% |
| KNN                 |   83.90% |    66.80% | 41.52% |   51.21% |  78.28% |
| SVM                 |   86.25% |    84.74% | 39.56% |   53.94% |  82.31% |
| XGBoost             |   86.45% |    75.76% | 49.14% |   59.61% |  85.61% |

---

## 🔧 Hyperparameter Tuning

Random Forest was further optimized using **GridSearchCV**.

### Best Parameters

```text
n_estimators = 100
max_depth = 15
min_samples_split = 5
```

### Tuned Random Forest Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **86.65%** |
| Precision | **79.17%** |
| Recall    | **46.68%** |
| F1 Score  | **58.73%** |
| ROC AUC   | **85.54%** |

The tuned Random Forest achieved a cross-validation accuracy of approximately **86.49%** during hyperparameter search.

---

## 🧠 Why Random Forest?

Although the initial XGBoost model achieved slightly higher test accuracy, Random Forest was selected after hyperparameter tuning because the tuned model achieved:

* Improved test accuracy
* Strong precision
* Competitive ROC-AUC
* Stable cross-validation performance
* Good overall performance for this classification problem

This model was then saved and used by the FastAPI application.

---

## ⚡ FastAPI REST API

The trained model is exposed through a REST API using FastAPI.

### Endpoint

```text
POST /predict
```

### Example Request

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 45,
  "Tenure": 5,
  "Balance": 120000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "EstimatedSalary": 95000
}
```

### Example Response

```json
{
  "prediction": "Customer Will Not Churn",
  "churn_probability": "23.77%",
  "confidence": "76.23%",
  "model": "Random Forest (Tuned)",
  "status": "success"
}
```

### API Documentation

The interactive Swagger documentation is available at:

https://customer-churn-prediction-d4i0.onrender.com/docs

---

## 🐳 Docker

The application is containerized using Docker to provide a consistent runtime environment.

### Build Docker Image

```bash
docker build -t customer-churn-api .
```

### Run Container

```bash
docker run -p 8000:8000 customer-churn-api
```

### Open Swagger

```text
http://localhost:8000/docs
```

---

## ☁️ Deployment

The Dockerized FastAPI application is deployed on **Render**.

### Deployment Architecture

```text
GitHub Repository
       ↓
     Render
       ↓
 Docker Build
       ↓
 FastAPI Application
       ↓
   Uvicorn Server
       ↓
 Public REST API
```

### Live API

https://customer-churn-prediction-d4i0.onrender.com/docs

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
│
├── data/
│   └── Churn_Modelling.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── Customer_Churn_EDA.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── train.py
│   └── tune_model.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 💻 Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model

```bash
python src/train.py
```

### 6. Start FastAPI

```bash
uvicorn app.main:app --reload
```

### 7. Open API Documentation

```text
http://127.0.0.1:8000/docs
```

---

## 📈 Key Results

* Dataset size: **10,000 records**
* Features used: **10**
* Models compared: **6**
* Best tuned model: **Random Forest**
* Tuned accuracy: **86.65%**
* Tuned ROC-AUC: **85.54%**
* Cross-validation accuracy during tuning: **86.49%**
* REST API: **FastAPI**
* Containerization: **Docker**
* Cloud deployment: **Render**

---

## 🔮 Future Improvements

* Address class imbalance using techniques such as class weights or SMOTE
* Optimize the classification threshold to improve churn recall
* Add explainable AI using SHAP
* Add automated CI/CD using GitHub Actions
* Add monitoring and logging for the deployed API
* Build a frontend dashboard using Streamlit
* Add automated model retraining

---

## 👨‍💻 Author

**Linga Raju Potu**

B.Tech — Artificial Intelligence & Machine Learning

Interested in Machine Learning, Generative AI, Python Development, and AI-powered applications.
