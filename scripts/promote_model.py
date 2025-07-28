<<<<<<< Updated upstream
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
=======
import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up MLflow tracking URI for DagsHub
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Roopendra-M"
        repo_name = "Capstone_Project"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.client = MlflowClient()

        # Find the latest run for the given experiment name
        experiment_name = "Default"
        experiment = cls.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise Exception(f"Experiment '{experiment_name}' not found.")

        runs = cls.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        assert runs, "No runs found in the experiment."

        cls.latest_run_id = runs[0].info.run_id
        cls.model_uri = f"runs:/{cls.latest_run_id}/model"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        input_text = "Hi how are you..."
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        prediction = self.model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.model.predict(X_holdout)

        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred)

        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy, expected_accuracy, f"Accuracy should be ≥ {expected_accuracy}")
        self.assertGreaterEqual(precision, expected_precision, f"Precision should be ≥ {expected_precision}")
        self.assertGreaterEqual(recall, expected_recall, f"Recall should be ≥ {expected_recall}")
        self.assertGreaterEqual(f1, expected_f1, f"F1 should be ≥ {expected_f1}")


if __name__ == "__main__":
    unittest.main()
>>>>>>> Stashed changes
