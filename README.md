# 🚗 Vehicle Insurance Claim Prediction - MLOps Capstone Project

A Machine Learning project designed to predict vehicle insurance claim likelihood using customer features. This end-to-end pipeline demonstrates MLOps capabilities like DVC, MLflow, Docker, Kubernetes (Minikube), GitHub Actions CI/CD, and Prometheus + Grafana monitoring.

---

## 🧠 Problem Statement

Predict whether a customer is likely to make a vehicle insurance claim in the next year based on demographic and driving profile data.

---

## 🔍 Sentiment Analysis (Streamlit App Feature)

The project also includes a simple **Sentiment Analysis UI** using Streamlit for capturing and analyzing customer feedback. Text entered by users is classified into:
- Positive
- Negative
- Neutral

This is integrated for exploratory purposes and not part of the core ML pipeline.

---

## ⚙️ Project Structure

```bash
Capstone_Project/
├── data/                     # Raw and processed data (DVC-tracked)
├── models/                   # Trained model files (tracked with DVC)
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── register_model.py
├── app/
│   └── main.py               # Streamlit/Dash web app
├── Dockerfile
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── prometheus-config.yaml
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI workflow
├── dvc.yaml
├── README.md
└── requirements.txt
🔄 CI/CD Pipeline (GitHub Actions)
✅ Continuous Integration

Unit testing on push/PR

Linting, formatting check

Auto data version check with DVC

✅ Continuous Delivery

Model training & registration (MLflow)

Model versioning with DVC

Push model & artifacts to GitHub and DVC remote

Sample: .github/workflows/ci.yml

yaml
Copy
Edit
- name: Checkout repo
- name: Setup Python
- name: Install deps
- name: Run DVC
- name: Train and evaluate
📦 Model Versioning with DVC
Track large files like data and models:

bash
Copy
Edit
dvc init
dvc add data/train.csv
dvc add models/model.pkl
git add data/.gitignore models/.gitignore
git commit -m "Track data and model with DVC"
Push to remote:

bash
Copy
Edit
dvc remote add -d myremote /path/to/local/remote
dvc push
🐳 Dockerization
🔧 Build the Docker image

<img width="1920" height="1080" alt="docker" src="https://github.com/user-attachments/assets/6a9376a2-6a01-4b2f-99d2-05bf1504fe6e" />



bash
Copy
Edit
docker build -t roopendra/vehicle-insurance:latest .
☁️ Push to Docker Hub


<img width="1920" height="1080" alt="Dockerhub" src="https://github.com/user-attachments/assets/6a2aa060-0822-4547-8c64-e138ede15083" />



Copy
Edit
docker login
docker push roopendra/vehicle-insurance:latest
☸️ Kubernetes (Minikube)
🔁 Start Minikube

bash
Copy
Edit
minikube start
🛠️ Deploy Application

bash
Copy
Edit
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
🌐 Access the App

bash
Copy
Edit
minikube service vehicle-insurance-service
📊 Monitoring with Prometheus & Grafana
📥 Install Prometheus and Grafana

bash
Copy
Edit
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
🔧 Add custom scrape config (example)

Edit prometheus-config.yaml:

yaml
Copy
Edit
scrape_configs:
  - job_name: 'capstone_project'
    static_configs:
      - targets: ['<minikube_ip>:<your_app_port>']
Apply config:

bash
Copy
Edit
kubectl apply -f prometheus-config.yaml
📈 Access Dashboards


<img width="1920" height="1080" alt="prometheus" src="https://github.com/user-attachments/assets/3c0ca0cd-003a-4e11-87e8-2d08a736eb45" />


bash
Copy
Edit
kubectl port-forward svc/monitoring-grafana -n monitoring 3000:80
Visit: http://localhost:3000
Login: admin / prom-operator

<img width="1920" height="1080" alt="grafana" src="https://github.com/user-attachments/assets/1f0a72ed-87c3-4777-909b-670cb8e0bef1" />

<img width="1920" height="1080" alt="grafana1" src="https://github.com/user-attachments/assets/64442a40-86a1-4d52-b0d7-378c6a4c3427" />

👨‍💻 Author
Roopendra R
B.Tech CSE, RGUKT RK Valley
📫 Email: mardalaroopendra@gmail.com
🌐 LinkedIn: linkedin.com/in/roopendra-r

🏁 Conclusion
This project demonstrates the practical implementation of a full ML lifecycle pipeline using modern DevOps tools. It combines machine learning, model reproducibility, cloud-native tools (K8s, Docker), and open-source observability.

⭐ Star this repo if you found it useful!
