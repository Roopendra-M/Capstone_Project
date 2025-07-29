🔍 Sentiment Analysis- MLOps Capstone Project

A Machine Learning project designed to analysis the sentiment usinf text. This end-to-end pipeline demonstrates MLOps capabilities like **DVC**, **MLflow**, **Docker**, **Kubernetes (Minikube)**, **GitHub Actions CI/CD**, and **Prometheus + Grafana** monitoring.

---

## 🧠 Problem Statement

Predict whether a customer is likely to make a vehicle insurance claim in the next year based on demographic and driving profile data.

---

## 🔍 Sentiment Analysis

The project also includes a simple **Sentiment Analysis UI** using Streamlit for capturing and analyzing customer feedback.  
Text entered by users is classified into:
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

Linting & formatting checks

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

![docker](https://raw.githubusercontent.com/Roopendra-M/Capstone_Project/main/references/docker.png)


bash
Copy
Edit
docker build -t roopendra/vehicle-insurance:latest .
☁️ Push to Docker Hub


![docker hub](https://raw.githubusercontent.com/Roopendra-M/Capstone_Project/main/references/Dockerhub.png)

bash
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
![prometheus](https://raw.githubusercontent.com/Roopendra-M/Capstone_Project/main/references/prometheus.png)
bash
Copy
Edit
kubectl apply -f prometheus-config.yaml
📈 Access Dashboards



bash
Copy
Edit
kubectl port-forward svc/monitoring-grafana -n monitoring 3000:80
Visit: http://localhost:3000
Login: admin / prom-operator

![dashboard](https://raw.githubusercontent.com/Roopendra-M/Capstone_Project/main/references/grafana.png)
![dashboard1](https://raw.githubusercontent.com/Roopendra-M/Capstone_Project/main/references/grafana1.png)





👨‍💻 Author
Roopendra R
B.Tech CSE, RGUKT RK Valley
📫 Email: mardalaroopendra@gmail.com
🏁 Conclusion
This project demonstrates the practical implementation of a full ML lifecycle pipeline using modern DevOps tools. It combines machine learning, model reproducibility, cloud-native tools (K8s, Docker), and open-source observability.

⭐ Star this repo if you found it useful!
