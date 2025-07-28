import os
import mlflow
from mlflow.exceptions import RestException
import dagshub

# Set DagsHub credentials
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "Roopendra-M"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Roopendra-M"
repo_name = "Capstone_Project"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def promote_candidate_to_production(model_name="My_model"):
    client = mlflow.MlflowClient()

    # Get candidate version
    try:
        candidate_version = client.get_model_version_by_alias(model_name, "candidate").version
        print(f"Found candidate version: {candidate_version}")
    except RestException:
        print(" No candidate model found.")
        return

    # Remove 'production' from any version
    try:
        current_prod = client.get_model_version_by_alias(model_name, "production")
        client.delete_model_version_alias(model_name, "production", current_prod.version)
        print(f" Removed existing production alias from version {current_prod.version}")
    except RestException:
        print(" No previous production alias to remove.")

    # Assign 'production' alias to candidate
    client.set_model_version_alias(model_name, "production", candidate_version)
    print(f" Promoted version {candidate_version} to alias 'production'")

if __name__ == "__main__":
    promote_candidate_to_production()
