import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Roopendra-M"
        repo_name = "Capstone_Project"

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        cls.model_name = "My_model"
        cls.latest_version = cls.get_latest_version_number(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.latest_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_version_number(model_name):
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        return versions[0].version if versions else None

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

        self.assertGreaterEqual(accuracy, 0.40, "Accuracy should be at least 0.40")
        self.assertGreaterEqual(precision, 0.40, "Precision should be at least 0.40")
        self.assertGreaterEqual(recall, 0.40, "Recall should be at least 0.40")
        self.assertGreaterEqual(f1, 0.40, "F1 Score should be at least 0.40")


if __name__ == "__main__":
    unittest.main()
