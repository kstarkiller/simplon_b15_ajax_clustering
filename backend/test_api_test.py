import unittest
from endpoints import pca
from endpoints import kmeans_and_gmm

class TestPCAPredict(unittest.TestCase):
    def test_pca_predict(self):
        result = pca.pca_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIn("MSE", result)
        self.assertIn("plot_base64", result)

    def test_pca_predict_returns_valid_data(self):
        result = pca.pca_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], list)
        self.assertNotEqual(len(result["data"]), 0)
        for data_point in result["data"]:
            self.assertIsInstance(data_point, dict)
            self.assertIn("x", data_point)
            self.assertIn("y", data_point)

    def test_pca_predict_returns_valid_MSE(self):
        result = pca.pca_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("MSE", result)
        self.assertIsInstance(result["MSE"], float)
        self.assertGreaterEqual(result["MSE"], 0)

    def test_pca_predict_returns_valid_plot_base64(self):
        result = pca.pca_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("plot_base64", result)
        self.assertIsInstance(result["plot_base64"], str)
        self.assertNotEqual(result["plot_base64"], "")

if __name__ == '__main__':
    unittest.main()