# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")


# set up for mlflow tracking url

# mlflow.set_tracking_uri("https://dagshub.com/Roopendra-M/Capstone_Project.mlflow")
# dagshub.init(repo_owner="Roopendra-M",repo_name="Capstone_Project",mlflow=True)
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Roopendra-M"
repo_name = "Capstone_Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def load_model_info(file_path:str)->dict:
    """Load the model information from a JSON file"""
    try:
        with open(file_path,'r') as file:
            model_info=json.load(file)
        logging.info("Model loaded from the %s",file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found %s",file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading the model information : %s",e)
        raise

def register_model(model_name:str, model_info:dict) -> None:
    """Register the model to the MLflow model registry and assign alias 'candidate'."""
    try:
        model_url = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_url, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="None",  # Since we're using alias, we can skip formal staging
            archive_existing_versions=True
        )

        # Assign alias
        client.set_registered_model_alias(
            name=model_name,
            version=model_version.version,
            alias="candidate"
        )

        logging.info(f"Model {model_name} version {model_version.version} registered with alias 'candidate'")
        print(f"Model {model_name} version {model_version.version} assigned alias 'candidate'.")

    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info_path='reports/experiment_info.json'
        model_info=load_model_info(model_info_path)

        model_name="My_model"
        register_model(model_name,model_info)
    except Exception as e:
        logging.error("Failed to complete the model registration process %s",e)
        print(f"Error : {e}")

    
if __name__=="__main__":
    main()
        