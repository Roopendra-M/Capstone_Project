# 🔍 Sentiment Analysis – MLOps Capstone Project

A **production-grade Machine Learning pipeline** for sentiment analysis and insurance claim risk prediction.  
This project demonstrates **end-to-end MLOps** using:

**DVC** • **MLflow** • **Docker** • **Kubernetes (Minikube)** • **GitHub Actions CI/CD** • **Prometheus** • **Grafana**

---

## 🧠 Problem Statement

1. **Insurance Claim Prediction** – Predict whether a customer will make a vehicle insurance claim in the next year based on demographic and driving profile data.  
2. **Sentiment Analysis (UI)** – Capture and analyze customer feedback through a **Streamlit** UI to classify text as:
   - Positive  
   - Negative  
   - Neutral  

> Sentiment Analysis is included for exploratory purposes and not part of the core insurance pipeline.

---

## 📌 Features

✅ **Data & Model Versioning** with DVC  
✅ **Automated CI/CD** using GitHub Actions  
✅ **Model Registry** with MLflow  
✅ **Containerization** using Docker  
✅ **Scalable Deployment** on Kubernetes (Minikube)  
✅ **Real-time Monitoring** with Prometheus & Grafana  
✅ **Interactive UI** for sentiment analysis

---

## ⚙️ Project Structure

Capstone_Project/
├── data/ # Raw & processed data (DVC-tracked)
├── models/ # Trained model files (DVC-tracked)
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── register_model.py
├── app/
│ └── main.py # Streamlit app
├── Dockerfile
├── kubernetes/
│ ├── deployment.yaml
│ ├── service.yaml
│ └── prometheus-config.yaml
├── .github/workflows/
│ └── ci.yml # GitHub Actions CI workflow
├── dvc.yaml
└── requirements.txt

yaml
Copy
Edit

---

## 🚀 MLOps Workflow

### 🔄 CI/CD Pipeline (GitHub Actions)
- **Continuous Integration**
  - Unit testing on push/PR
  - Linting & formatting checks
  - Auto data version check with DVC
- **Continuous Delivery**
  - Model training & registration (MLflow)
  - Model versioning with DVC
  - Push artifacts to GitHub & DVC remote

[View CI Workflow →](.github/workflows/ci.yml)

---

### 📦 Model Versioning with DVC
```bash
dvc init
dvc add data/train.csv
dvc add models/model.pkl
git add data/.gitignore models/.gitignore
git commit -m "Track data and model with DVC"
dvc remote add -d myremote /path/to/local/remote
dvc push
🐳 Dockerization
bash
Copy
Edit
docker build -t roopendra/vehicle-insurance:latest .
docker push roopendra/vehicle-insurance:latest



☸ Kubernetes Deployment
bash
Copy
Edit
minikube start
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
minikube service vehicle-insurance-service
📊 Monitoring with Prometheus & Grafana
bash
Copy
Edit
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
kubectl apply -f kubernetes/prometheus-config.yaml
kubectl port-forward svc/monitoring-grafana -n monitoring 3000:80
Visit → http://localhost:3000 (user: admin, pass: prom-operator)





🛠️ Tech Stack
Languages: Python

ML Frameworks: scikit-learn, TensorFlow, Hugging Face

MLOps Tools: DVC, MLflow, GitHub Actions, Docker, Kubernetes

Monitoring: Prometheus, Grafana

Deployment: Streamlit, FastAPI

👨‍💻 Author
Roopendra R
B.Tech CSE, RGUKT RK Valley
📧 Email: mardalaroopendra@gmail.com
🔗 GitHub: Roopendra-M

🏁 Conclusion
This project showcases a full ML lifecycle pipeline integrating machine learning, reproducibility, cloud-native tools (K8s, Docker), and open-source observability tools for a truly production-ready AI workflow.

⭐ If you found this useful, consider starring the repo!
