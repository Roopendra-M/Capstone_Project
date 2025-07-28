import os
import json
import mlflow
from src.logger import logging  # make sure you have this logger or replace with print
import warnings
from mlflow.exceptions import RestException
import dagshub

<<<<<<< Updated upstream
warnings.filterwarnings("ignore")

# Set up DagsHub credentials
=======
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set up DagsHub MLflow tracking
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info("Model loaded from the %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    try:
        model_url = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_url, model_name)

        logging.info(f"Model {model_name} version {model_version.version} registered.")
        print(f"Model {model_name} version {model_version.version} registered successfully.")

    except Exception as e:
        logging.error("Error during model registration: %s", e)
>>>>>>> Stashed changes
        raise


def main():
<<<<<<< Updated upstream
    model_info_path = "reports/experiment_info.json"
    model_info = load_model_info(model_info_path)
    model_name = "My_model"
    register_and_alias_model(model_name, model_info)

=======
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "My_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


>>>>>>> Stashed changes
if __name__ == "__main__":
    main()
