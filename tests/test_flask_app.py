import unittest
from flask_app.app import app  # Ensure this path is correct

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<title>Sentiment Analysis</title>", response.data)

    def test_predict_page(self):
        response = self.client.post("/predict", data=dict(text="I Love this !"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b"Positive" in response.data or b"Negative" in response.data,
            "Response should contain either 'Positive' or 'Negative'" 
        )

if __name__ == "__main__":
    unittest.main()
