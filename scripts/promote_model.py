# promote_model.py
import os
import mlflow

def promote_model():
    # Fetch DagsHub token from environment
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    # Set token for MLflow tracking
    os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token

    # DagsHub MLflow Tracking URI
    dagshub_url = "https://dagshub.com"
    repo_owner = "Roopendra-M"
    repo_name = "Capstone_Project"
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MlflowClient()
    model_name = "My_model"

    # Get the model version with alias "candidate"
    candidate_versions = client.get_model_version_by_alias(model_name, "candidate")
    candidate_version = candidate_versions.version

    print(f" Candidate model version: {candidate_version}")

    # Remove alias "production" from any currently promoted version
    try:
        current_prod_version = client.get_model_version_by_alias(model_name, "production")
        client.delete_model_version_alias(model_name, "production", current_prod_version.version)
        print(f" Removed 'production' alias from version {current_prod_version.version}")
    except mlflow.exceptions.RestException:
        print("ℹ No existing 'production' model to unassign.")

    # Promote candidate model to production
    client.set_model_version_alias(model_name, "production", candidate_version)
    print(f" Promoted version {candidate_version} to alias 'production'")

if __name__ == "__main__":
    promote_model()
