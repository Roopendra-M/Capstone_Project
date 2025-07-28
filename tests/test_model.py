# testing_model_loading.py
import unittest
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "My_model"  # <-- change to your actual model name
        cls.alias = "candidate"      # <-- test the model assigned to 'candidate'
        cls.client = MlflowClient()
        cls.model_uri = cls.get_model_uri_by_alias(cls.model_name, cls.alias)
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

    @staticmethod
    def get_model_uri_by_alias(model_name, alias):
        """
        Retrieves the URI of the model version assigned to a given alias.
        """
        client = MlflowClient()
        try:
            version = client.get_model_version_by_alias(model_name, alias)
            model_uri = f"models:/{model_name}/{version.version}"
            return model_uri
        except Exception as e:
            raise RuntimeError(f"Error retrieving model URI for alias '{alias}': {e}")

    def test_model_loading(self):
        """Verifies that the model loads correctly."""
        self.assertIsNotNone(self.model, "Model failed to load.")

    def test_model_prediction(self):
        """Verifies that the model can return predictions on sample input."""
        sample_input = ["I love this product!"]  # adjust for your model input
        prediction = self.model.predict(sample_input)

        self.assertIsNotNone(prediction, "Model prediction returned None.")
        self.assertTrue(len(prediction) > 0, "Prediction is empty.")

if __name__ == "__main__":
    unittest.main()
