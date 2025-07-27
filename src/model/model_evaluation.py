import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging


mlflow.set_tracking_uri("https://dagshub.com/Roopendra-M/Capstone_Project.mlflow")
dagshub.init(repo_owner="Roopendra-M",repo_name="Capstone_Project",mlflow=True)


def load_data(file_path:str)-> pd.DataFrame:
    """Loads the data from the csv file...."""
    try:
        df=pd.read_csv(file_path)
        logging.info("Data loaded from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the csv file : %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading the data : %s",e)
        raise

def load_model(file_path):
    """Loads the model from the file"""
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
        logging.info("Model loaded from %s",file_path)
        return model
    except FileNotFoundError as e:
        logging.error("File not found : %s",file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading the model : %s",e)
        raise

def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)-> dict:
    """Evaluate the model and returns the evaluation metrics...."""
    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred)

        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logging.info("Model Evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logging.error("Error during model evaluation : %s",e)
        raise
def save_metrics(metrics:dict,file_path:str)->None:
    """Save the evaluation matrics to a json file"""
    try:
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logging.info("Metrics information saved to %s",file_path)
    except Exception as e:
        logging.error("Error occured while saving the metrics information : %s",e)
        raise

def save_model_info(run_id:str,model_path:str,file_path:str)-> None:
    """Save the mode run id and path to a json file..."""
    try:
        model_info={'run_id':run_id,'model_path':model_path}
        with open(file_path,'w') as file:
            json.dump(model_info,file,indent=4)
        logging.debug("Model info saved to %s",file_path)
    except Exception as e:
        logging.error("Error occured while saving the model information : %s",e)
        raise



def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf=load_model("./models/model.pkl")
            test_data=load_data("./data/processed/test_bow.csv")

            X_test=test_data.iloc[:,:-1].values
            y_test=test_data.iloc[:,-1].values

            metrics=evaluate_model(clf,X_test,y_test)
            save_metrics(metrics,'reports/metrics.json')

            # log model parameters to mlflow
            for metric_name,metric_value in metrics.items():
                mlflow.log_metric(metric_name,metric_value)
            
            # log parameters to the mlflow
            if hasattr(clf,'get_params'):
                params=clf.get_params()
                for param_name,param_value in params.items():
                    mlflow.log_param(param_name,param_value)

            # log model to mlflow
            mlflow.sklearn.log_model(clf,"model")

            save_model_info(run.info.run_id,"model","reports/experiment_info.json")
            # log the metrics files to mlflow

            mlflow.log_artifact('reports/metrics.json')
            logging.info("All parameters are mentioned in mlflow")
        except Exception as e:
            logging.error("Failed to complete the model evaluation process : %s",e)
            print(f"Error : {e}")
        

if __name__=="__main__":
    main()