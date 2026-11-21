
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import prepare_text

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "jobguard_model.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"


class TestJobGuardModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Trained model not found")

        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError("Threshold file not found")

        cls.model = joblib.load(MODEL_PATH)

        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            cls.threshold = float(json.load(f)["threshold"])

    def test_model_loading(self):
        self.assertIsNotNone(self.model)

    def test_threshold_validity(self):
        self.assertGreaterEqual(self.threshold, 0)
        self.assertLessEqual(self.threshold, 1)

    def test_valid_prediction(self):
        data = pd.DataFrame([{
            "title": "Software Developer",
            "company_profile": "ABC Technologies",
            "description": "We are hiring a Python developer.",
            "requirements": "Python and SQL knowledge",
            "benefits": "Health insurance"
        }])

        text = prepare_text(data)
        probabilities = self.model.predict_proba(text)

        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(
            np.all((probabilities >= 0) & (probabilities <= 1))
        )
        self.assertAlmostEqual(probabilities.sum(), 1.0)

    def test_missing_columns(self):
        data = pd.DataFrame([{"title": "Data Analyst"}])

        text = prepare_text(data)
        probabilities = self.model.predict_proba(text)

        self.assertEqual(probabilities.shape, (1, 2))

    def test_multiple_predictions(self):
        data = pd.DataFrame([
            {"title": "Python Developer"},
            {"title": "Marketing Intern"},
            {"title": "Data Analyst"}
        ])

        probabilities = self.model.predict_proba(
            prepare_text(data)
        )

        self.assertEqual(probabilities.shape, (3, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
  
