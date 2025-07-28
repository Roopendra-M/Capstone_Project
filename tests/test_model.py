import unittest
import mlflow
<<<<<<< Updated upstream
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
=======
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
>>>>>>> Stashed changes

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
<<<<<<< Updated upstream
        #  Set the MLflow tracking URI for DagsHub
        mlflow.set_tracking_uri("https://dagshub.com/Roopendra-M/Capstone_Project/mlflow")
=======
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
>>>>>>> Stashed changes

        cls.model_name = "My_model"
        cls.alias = "candidate"  # Or "production"
        cls.client = MlflowClient()

<<<<<<< Updated upstream
        try:
            cls.model_uri = cls.get_model_uri_by_alias(cls.model_name, cls.alias)
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        except Exception as e:
            cls.model = None
            print(f"WARNING: Failed to load model with alias '{cls.alias}': {e}")

    @classmethod
    def get_model_uri_by_alias(cls, model_name, alias):
        try:
            version = cls.client.get_model_version_by_alias(model_name, alias)
            return f"models:/{model_name}/{version.version}"
        except Exception as e:
            raise RuntimeError(f"Error retrieving model URI for alias '{alias}': {e}")

    def test_model_loading(self):
        if self.model is None:
            self.skipTest(f"Model with alias '{self.alias}' not available.")
        self.assertIsNotNone(self.model, "Model failed to load.")

    def test_model_prediction(self):
        if self.model is None:
            self.skipTest(f"Model with alias '{self.alias}' not available.")

        sample_input = ["This is a great product!"]
        prediction = self.model.predict(sample_input)

        self.assertIsNotNone(prediction, "Model prediction returned None.")
        self.assertGreater(len(prediction), 0, "Prediction is empty.")
=======
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

>>>>>>> Stashed changes

if __name__ == "__main__":
    unittest.main()
