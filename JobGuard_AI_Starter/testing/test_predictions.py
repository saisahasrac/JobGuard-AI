
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


class TestJobPredictions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Trained model not found")

        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError("Threshold file not found")

        cls.model = joblib.load(MODEL_PATH)

        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            cls.threshold = float(json.load(f)["threshold"])

    def get_prediction(self, data):
        text = prepare_text(pd.DataFrame([data]))
        score = float(self.model.predict_proba(text)[0, 1])

        prediction = int(score >= self.threshold)

        return score, prediction

    def test_genuine_style_posting(self):
        data = {
            "title": "Software Developer",
            "company_profile": "Established technology company",
            "description": "We are hiring a Python developer.",
            "requirements": "Python, SQL and communication skills",
            "benefits": "Health insurance and paid leave"
        }

        score, prediction = self.get_prediction(data)

        self.assertTrue(0 <= score <= 1)
        self.assertIn(prediction, [0, 1])

    def test_suspicious_style_posting(self):
        data = {
            "title": "Work From Home",
            "description": "Earn money instantly with no experience. "
                           "Pay a registration fee to get started."
        }

        score, prediction = self.get_prediction(data)

        self.assertTrue(0 <= score <= 1)
        self.assertIn(prediction, [0, 1])

    def test_missing_values(self):
        data = {
            "title": "Data Analyst",
            "description": None,
            "company_profile": np.nan,
            "requirements": "Python knowledge"
        }

        score, prediction = self.get_prediction(data)

        self.assertTrue(np.isfinite(score))
        self.assertIn(prediction, [0, 1])

    def test_short_description(self):
        data = {
            "title": "Intern",
            "description": "Apply now"
        }

        score, prediction = self.get_prediction(data)

        self.assertTrue(np.isfinite(score))
        self.assertIn(prediction, [0, 1])

    def test_empty_text_fields(self):
        data = {
            "title": "",
            "description": "",
            "company_profile": "",
            "requirements": ""
        }

        score, prediction = self.get_prediction(data)

        self.assertTrue(np.isfinite(score))
        self.assertIn(prediction, [0, 1])

    def test_threshold_classification(self):
        data = {
            "title": "Marketing Intern",
            "description": "Marketing internship opportunity"
        }

        score, prediction = self.get_prediction(data)

        expected = int(score >= self.threshold)

        self.assertEqual(prediction, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
  
