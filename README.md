# Iris Classifier API Deployment 🚀

## 📌 Overview
This project demonstrates the end-to-end deployment of a Machine Learning model using FastAPI, Docker, and Google Cloud Run.

The application predicts the species of an Iris flower based on input features.

---

## 🧠 Model
- Algorithm: Random Forest Classifier  
- Dataset: Iris Dataset (scikit-learn)  
- Output: Predicted species with confidence score  

---

## ⚙️ Technologies Used
- Python  
- FastAPI  
- Scikit-learn  
- Docker  
- Google Cloud Run  
- GCP Artifact Registry  

---

 📂 Project Structure
```bash
iris-deploy/
│
├── app/
│ ├── main.py
│ └── iris_model.pkl
│
├── train_model.py
├── requirements.txt
├── Dockerfile
```
---

## 🚀 How to Run Locally

### 1. Install Dependencies

pip install -r requirements.txt


### 2. Train Model

python train_model.py


### 3. Run FastAPI

uvicorn app.main:app --reload


### 4. Open in Browser

http://localhost:8000/docs


---

## 🐳 Docker Setup

### Build Image

docker build -t iris-api:v1 .


### Run Container

docker run -p 8080:8080 iris-api:v1


---

## ☁️ Deployment (GCP Cloud Run)

The application was deployed using:
- Docker container
- GCP Artifact Registry
- Cloud Run service

### 🌐 Live URL

https://iris-api-44849949642.us-central1.run.app


---

## 🔍 API Endpoints

### Health Check

GET /health


### Prediction

POST /predict


Example Input:
```json
{
  "sepal_length": 6.3,
  "sepal_width": 3.3,
  "petal_length": 6.0,
  "petal_width": 2.5
}
🧪 Testing
Health
curl https://iris-api-44849949642.us-central1.run.app/health
Predict

Use Swagger UI:

https://iris-api-44849949642.us-central1.run.app/docs
🧹 Cleanup

To avoid charges, the deployed service and resources were deleted after completion.

📌 Note

This project was developed as part of a hands-on lab to understand real-world ML deployment using containerization and cloud services.
