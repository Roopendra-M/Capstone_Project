import os
import mlflow

def promote_model():
    # Fetch DagsHub token from environment
    dagshub_token = os.getenv("CAPSTONE_TEST")  # This should be the personal access token (PAT)
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    # Set credentials for MLflow tracking
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Roopendra-M"   # Your DagsHub username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token   # Your PAT

    # DagsHub MLflow Tracking URI
    dagshub_url = "https://dagshub.com"
    repo_owner = "Roopendra-M"
    repo_name = "Capstone_Project"
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MlflowClient()
    model_name = "My_model"

    # Get the model version with alias "candidate"
    candidate_version = client.get_model_version_by_alias(model_name, "candidate").version
    print(f" Candidate model version: {candidate_version}")

    # Remove alias "production" from any current version
    try:
        current_prod_version = client.get_model_version_by_alias(model_name, "production")
        client.delete_model_version_alias(model_name, "production", current_prod_version.version)
        print(f" Removed 'production' alias from version {current_prod_version.version}")
    except mlflow.exceptions.RestException:
        print("ℹ No existing 'production' model to unassign.")

    # Promote candidate to production
    client.set_model_version_alias(model_name, "production", candidate_version)
    print(f" Promoted version {candidate_version} to alias 'production'")

if __name__ == "__main__":
    promote_model()
