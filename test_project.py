"""Behavioral verification: python -m unittest -v test_project.py."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.config import BASE, AS_OF, FEATURES, FREQUENCY, MONETARY, ALGORITHMS, BUNDLE
from src.data_generation import generate_data
from src.preprocessing import compute_rfm, prepare_customers
from src.clustering import evaluate_clustering, rank_models
from src.workflow import analyze, load_bundle, save_bundle
from src.prediction import predict_customer
from src.database import store_dataset, load_customers, store_results, connect


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers, cls.transactions = generate_data(200)
        cls.bundle = analyze(cls.customers, cls.transactions, k=4)

    def test_ledger_reconciles(self):
        data = compute_rfm(self.customers, self.transactions)
        self.assertEqual(int(data[FREQUENCY].sum()), len(self.transactions))
        self.assertAlmostEqual(data[MONETARY].sum(), self.transactions.Amount.sum(), places=5)
        expected = (pd.Timestamp(AS_OF) - pd.to_datetime(data["Last Purchase Date"])).dt.days
        np.testing.assert_array_equal(data.Recency, expected)

    def test_no_purchase_customer(self):
        transactions = self.transactions[self.transactions.CustomerID != 1]
        data = compute_rfm(self.customers, transactions)
        row = data[data.CustomerID == 1].iloc[0]
        self.assertEqual(row[FREQUENCY], 0)
        self.assertEqual(row[MONETARY], 0)
        self.assertEqual(row.Recency, 366)

    def test_invalid_dates_ids_and_numeric_values(self):
        duplicate = self.customers.copy()
        duplicate.loc[0, "CustomerID"] = duplicate.loc[1, "CustomerID"]
        with self.assertRaises(ValueError):
            prepare_customers(duplicate)
        invalid = self.customers.copy()
        invalid["Age"] = np.nan
        with self.assertRaises(ValueError):
            prepare_customers(invalid)
        tx = self.transactions.copy()
        tx.loc[0, "Purchase Date"] = "2099-01-01"
        with self.assertRaises(ValueError):
            compute_rfm(self.customers, tx)
        tx.loc[0, "Purchase Date"] = "not-a-date"
        with self.assertRaises(ValueError):
            compute_rfm(self.customers, tx)

    def test_imputation_and_roundtrip(self):
        data = self.customers.astype({"Age": float})
        data.loc[0, "Age"] = np.nan
        bundle = analyze(data, k=3)
        self.assertFalse(bundle["data"][FEATURES].isna().any().any())
        self.assertEqual(bundle["data"].loc[0, "Age"], data.Age.median())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save_bundle(bundle, path)
            restored = load_bundle(path)
            customer = bundle["data"].iloc[0].to_dict()
            for name in ALGORITHMS:
                self.assertEqual(predict_customer(bundle, customer, name)["cluster"], predict_customer(restored, customer, name)["cluster"])

    def test_native_prediction_matches_training_and_explanation(self):
        for name in ["K-Means", "Gaussian Mixture"]:
            row = self.bundle["data"].iloc[0].to_dict()
            result = predict_customer(self.bundle, row, name)
            self.assertEqual(result["cluster"], int(self.bundle["models"][name]["labels"][0]))
            self.assertTrue(result["category"])
            self.assertTrue(result["recommendation"])
        result = predict_customer(self.bundle, row, "K-Means")
        self.assertGreaterEqual(result["details"]["Support vs next center"].sum(), 0)

    def test_dbscan_unknown_and_hierarchical_proxy(self):
        row = self.bundle["data"].iloc[0].to_dict()
        row["Annual Income"] = 100000
        result = predict_customer(self.bundle, row, "DBSCAN")
        self.assertEqual(result["cluster"], -1)
        self.assertTrue(result["warnings"])
        self.assertIn("proxy", predict_customer(self.bundle, row, "Agglomerative")["method"])

    def test_invalid_partition_and_coverage_ranking(self):
        x = np.arange(30).reshape(10, 3)
        for labels in [np.zeros(10), -np.ones(10), np.arange(10)]:
            self.assertTrue(np.isnan(evaluate_clustering(x, labels)["Silhouette"]))
        table = pd.DataFrame([
            {"Model": "Noisy", "Coverage": .2, "Silhouette": .99, "Davies-Bouldin": .1, "Calinski-Harabasz": 1000},
            {"Model": "Complete", "Coverage": 1., "Silhouette": .5, "Davies-Bouldin": 1., "Calinski-Harabasz": 100},
        ])
        _, best = rank_models(table)
        self.assertEqual(best, "Complete")

    def test_sqlite_roundtrip_and_foreign_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            dataset_id = store_dataset(self.customers, self.transactions, AS_OF, "test", path)
            self.assertEqual(len(load_customers(dataset_id, path)), 200)
            run_id = store_results(self.bundle, dataset_id, path)
            connection = connect(path)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM cluster_results WHERE run_id=?", (run_id,)).fetchone()[0], 800)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            connection.close()

    def test_classic_compatibility(self):
        data = self.customers[["CustomerID", "Gender", "Age", "Annual Income", "Spending Score"]]
        data = data.rename(columns={"Annual Income": "Annual Income (k$)", "Spending Score": "Spending Score (1-100)"})
        bundle = analyze(data, mode="classic", k=4)
        self.assertEqual(len(bundle["features"]), 2)
        self.assertEqual(len(bundle["data"]), 200)

    def test_dashboard_pages_and_prediction(self):
        app = AppTest.from_file(str(BASE / "app.py"), default_timeout=60).run()
        self.assertTrue(any(element.value == "Administrator login" for element in app.subheader))
        app.text_input(key="login_username").set_value("admin")
        app.text_input(key="login_password").set_value("admin123")
        [button for button in app.button if button.label == "Sign in"][0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        for page in ["dataset_analysis", "customer_segmentation", "model_comparison", "marketing_recommendations", "customer_prediction"]:
            app.switch_page(f"app_pages/{page}.py").run()
            self.assertFalse(app.exception, page)
            self.assertFalse(app.error, page)
        for name in ALGORITHMS:
            app.selectbox(key="active_model").set_value(name).run()
            [button for button in app.button if button.label == "Predict customer segment"][0].click().run()
            self.assertFalse(app.exception, name)
            self.assertFalse(app.error, name)
            self.assertEqual(app.session_state.prediction["model"], name)
        [button for button in app.button if button.label == "Log out"][0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(element.value == "Administrator login" for element in app.subheader))
        self.assertNotIn("analytics", app.session_state)

    def test_authentication_sessions_and_lockout(self):
        from src.auth import authenticate, session_user, revoke_session
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.db"
            self.assertIsNone(authenticate("admin", "wrong", path, now=100))
            token = authenticate("admin", "admin123", path, now=101)
            self.assertEqual(session_user(token, path, now=102)["username"], "admin")
            self.assertIsNone(session_user("forged-token", path, now=103))
            revoke_session(token, path)
            self.assertIsNone(session_user(token, path, now=104))
            token = authenticate("admin", "admin123", path, now=105)
            self.assertIsNone(session_user(token, path, now=2000))
            for _ in range(5):
                self.assertIsNone(authenticate("admin", "wrong", path, now=3000))
            self.assertIsNone(authenticate("admin", "admin123", path, now=3001))
            self.assertIsNotNone(authenticate("admin", "admin123", path, now=3061))


if __name__ == "__main__":
    unittest.main()
