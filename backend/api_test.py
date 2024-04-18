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
        self.assertIsInstance(result["data"], dict)
        self.assertNotEqual(len(result["data"]), 0)
        for key, value in result["data"].items():
            self.assertIsInstance(value, dict)

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
        self.assertNotEqual(len(result["plot_base64"]), 0)


class TestIncomeKMeansPredict(unittest.TestCase):
    def test_income_kmeans_predict(self):
        result = kmeans_and_gmm.income_kmeans_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIn("plot_base64", result)
        self.assertIn("MSE", result)

    def test_income_kmeans_predict_returns_valid_data(self):
        result = kmeans_and_gmm.income_kmeans_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], dict)
        self.assertNotEqual(len(result["data"]), 0)
        for key, value in result["data"].items():
            self.assertIsInstance(value, dict)

    def test_income_kmeans_predict_returns_valid_plot_base64(self):
        result = kmeans_and_gmm.income_kmeans_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("plot_base64", result)
        self.assertIsInstance(result["plot_base64"], str)
        self.assertNotEqual(len(result["plot_base64"]), 0)

    def test_income_kmeans_predict_returns_valid_MSE(self):
        result = kmeans_and_gmm.income_kmeans_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("MSE", result)
        self.assertIsInstance(result["MSE"], float)
        self.assertGreaterEqual(result["MSE"], 0)


class TestIncomeGMMPredict(unittest.TestCase):
    def test_income_gmm_predict(self):
        result = kmeans_and_gmm.income_gmm_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIn("plot_base64", result)
        self.assertIn("AIC", result)

    def test_income_gmm_predict_returns_valid_data(self):
        result = kmeans_and_gmm.income_gmm_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], dict)
        self.assertNotEqual(len(result["data"]), 0)
        for key, value in result["data"].items():
            self.assertIsInstance(value, dict)

    def test_income_gmm_predict_returns_valid_plot_base64(self):
        result = kmeans_and_gmm.income_gmm_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("plot_base64", result)
        self.assertIsInstance(result["plot_base64"], str)
        self.assertNotEqual(len(result["plot_base64"]), 0)

    def test_income_gmm_predict_returns_valid_AIC(self):
        result = kmeans_and_gmm.income_gmm_predict()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("AIC", result)
        self.assertIsInstance(result["AIC"], float)
        self.assertGreaterEqual(result["AIC"], 0)


if __name__ == "__main__":
    unittest.main()
