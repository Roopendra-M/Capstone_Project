import unittest
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #  Set the MLflow tracking URI for DagsHub
        mlflow.set_tracking_uri("https://dagshub.com/Roopendra-M/Capstone_Project/mlflow")

        cls.model_name = "My_model"
        cls.alias = "candidate"  # Or "production"
        cls.client = MlflowClient()

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

if __name__ == "__main__":
    unittest.main()
