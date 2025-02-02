import unittest
import pandas as pd
from scripts.preprocess import load_data, preprocess_data

class TestPreprocess(unittest.TestCase):
    def test_load_data(self):
        df = load_data("data/sample_data.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_preprocess_data(self):
        df = load_data("data/sample_data.csv")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

if __name__ == "__main__":
    unittest.main()