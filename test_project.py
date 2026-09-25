"""Behavior checks: python -m unittest -v test_project.py."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from main import BASE, FEATURES, elbow_analysis, generate_dataset, preprocess, train_model


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            cls.raw = generate_dataset(Path(directory) / "customers.csv")

    def test_complete_workflow(self):
        data, _ = preprocess(self.raw)
        _, k = elbow_analysis(data)
        model, labeled, centers, score = train_model(data, k)
        self.assertEqual(len(labeled), 500)
        self.assertEqual(labeled.Cluster.nunique(), k)
        self.assertGreater(score, .4)
        np.testing.assert_array_equal(model.predict(data[FEATURES]), labeled.Cluster)
        np.testing.assert_allclose(centers.values, labeled.groupby("Cluster")[FEATURES].mean().values)

    def test_missing_values_preserve_customers(self):
        raw = self.raw.copy()
        raw.loc[0, FEATURES[0]] = np.nan
        clean, notes = preprocess(raw)
        self.assertEqual(len(clean), len(raw))
        self.assertEqual(clean.loc[0, FEATURES[0]], raw[FEATURES[0]].median())
        self.assertTrue(notes)

    def test_invalid_inputs(self):
        cases = []
        cases.append(self.raw.drop(columns="Age"))
        duplicate = self.raw.copy()
        duplicate.loc[0, "CustomerID"] = duplicate.loc[1, "CustomerID"]
        cases.append(duplicate)
        for value in [np.inf, -1, 101]:
            invalid = self.raw.astype({FEATURES[1]: float})
            invalid.loc[0, FEATURES[1]] = value
            cases.append(invalid)
        cases.append(self.raw.assign(**{FEATURES[0]: np.nan}))
        cases.append(self.raw.head(2))
        for raw in cases:
            with self.subTest(shape=raw.shape):
                with self.assertRaises(ValueError):
                    preprocess(raw)

    def test_dashboard_and_cluster_control(self):
        app = AppTest.from_file(str(BASE / "app.py")).run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        app.slider[0].set_value(3).run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[1].value, "3")


if __name__ == "__main__":
    unittest.main()
