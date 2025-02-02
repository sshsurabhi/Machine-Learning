import unittest
import torch
from scripts.train import SimpleNN, train_model
from scripts.preprocess import load_data, preprocess_data

class TestTrain(unittest.TestCase):
    def test_model_initialization(self):
        model = SimpleNN(input_size=4, hidden_size=10, output_size=1)
        self.assertIsInstance(model, SimpleNN)

    def test_train_model(self):
        df = load_data("data/sample_data.csv")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train, epochs=1)
        self.assertIsInstance(model, SimpleNN)

if __name__ == "__main__":
    unittest.main()