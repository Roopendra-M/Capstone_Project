import os
import json
import mlflow
from src.logger import logging  # make sure you have this logger or replace with print
import warnings
from mlflow.exceptions import RestException
import dagshub

warnings.filterwarnings("ignore")

# Set up DagsHub credentials
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "Roopendra-M"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI for DagsHub
dagshub_url = "https://dagshub.com"
repo_owner = "Roopendra-M"
repo_name = "Capstone_Project"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def load_model_info(file_path: str) -> dict:
    """Load model metadata from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f"Model info loaded from {file_path}")
        return model_info
    except Exception as e:
        logging.error(f"Error loading model info: {e}")
        raise

def register_and_alias_model(model_name: str, model_info: dict):
    try:
        client = mlflow.MlflowClient()
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        registered_model = mlflow.register_model(model_uri, model_name)

        # Optional: wait until registration completes
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="None",  # Staging not used
            archive_existing_versions=True
        )

        # Assign 'candidate' alias
        client.set_registered_model_alias(
            name=model_name,
            version=registered_model.version,
            alias="candidate"
        )

        logging.info(f"Model {model_name} version {registered_model.version} assigned alias 'candidate'")
        print(f"Registered {model_name} version {registered_model.version} with alias 'candidate'")

    except Exception as e:
        logging.error(f"Error during model registration: {e}")
        raise

def main():
    model_info_path = "reports/experiment_info.json"
    model_info = load_model_info(model_info_path)
    model_name = "My_model"
    register_and_alias_model(model_name, model_info)

if __name__ == "__main__":
    main()
