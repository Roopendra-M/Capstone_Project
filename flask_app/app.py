# Application for the Capstone Project
from flask import Flask,render_template,request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter,Histogram,generate_latest,CollectorRegistry,CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")
import numpy as np



def lemmatization(text):
    """Lemmatize the text"""
    lemmatizer=WordNetLemmatizer()
    text=text.split()
    text=[lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)
def remove_stop_words(text):
    """Remove the stop words from the text"""
    stop_words=set(stopwords.words("english"))
    text=[word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Removing the numbers from the text"""
    return " ".join([char for char in text if not char.isdigit()])

def lower_case(text):
    """Convert the text into lower case"""
    text=text.split()
    text=[word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Removes punctuations from the text"""
    text=re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    text=text.replace('؛',"")
    text=re.sub('\s+',' ',text).strip()
    return text

def removing_urls(text):
    """Removes URLs from the text"""
    url_pattern=re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'',text)


def normalize_text(text):
    text=lower_case(text)
    text=remove_stop_words(text)
    text=removing_numbers(text)
    text=removing_punctuations(text)
    text=removing_urls(text)
    text=lemmatization(text)

    return text


mlflow.set_tracking_uri("https://dagshub.com/Roopendra-M/Capstone_Project.mlflow")
dagshub.init(repo_owner="Roopendra-M",repo_name="Capstone_Project",mlflow=True)


app=Flask(__name__)

# from prometheus_client using CollectorRegistry

# Create a custom registry

registry=CollectorRegistry()
# Define custom metrics using this registry

REQUEST_COUNT=Counter(
    "app_request_count","Total number of requests to the app",["method","endpoint"],registry=registry
)
REQUEST_LATENCY=Histogram(
    "app_request_latency_seconds","Latency of requests in seconds",["endpoint"],registry=registry
)
PREDICTION_COUNT=Counter(
    "model_prediction_count","Count of predictions for each class ",["prediction"],registry=registry
)


# model and vectorizer setup
model_name="My_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    
    # Try fetching the Production stage version
    versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if versions:
        return versions[0].version

    # If no production version, try to get the latest among all versions
    all_versions = client.search_model_versions(f"name='{model_name}'")
    if all_versions:
        # Sort by version (as int), then get the latest
        latest = max(all_versions, key=lambda v: int(v.version))
        return latest.version

    return None


model_version=get_latest_model_version(model_name=model_name)
model_uri=f'models:/{model_name}/{model_version}'
print(f"Fetching model from : {model_uri}")
model=mlflow.pyfunc.load_model(model_uri)
vectorizer=pickle.load(open('models/vectorizer.pkl','rb'))

# Routes for the application of the Capstone Project.....

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET",endpoint="/").inc()
    start_time=time.time()
    response=render_template("index.html",result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time()-start_time)

    return response

@app.route("/predict",methods=["POST"])
def predict():
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
        start_time=time.time()
        
        text=request.form["text"]
        text=normalize_text(text)
        features=vectorizer.transform([text])
        features_df=pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
        result=model.predict(features_df)
        prediction=result[0]
        
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time()-start_time)

        return render_template("index.html", result=prediction)
    
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


@app.route("/metrics",methods=["GET"])
def metrics():
    """Expose only custom prometheus metrics"""
    return generate_latest(registry),200,{"Content-type":CONTENT_TYPE_LATEST}


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)