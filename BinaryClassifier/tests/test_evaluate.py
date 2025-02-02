import unittest
import torch
from scripts.evaluate import evaluate_model
from scripts.train import SimpleNN
from scripts.preprocess import load_data, preprocess_data

class TestEvaluate(unittest.TestCase):
    def test_evaluate_model(self):
        df = load_data("data/sample_data.csv")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = SimpleNN(input_size=X_train.shape[1], hidden_size=10, output_size=1)
        model.load_state_dict(torch.load("models/model.pth"))
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    unittest.main()