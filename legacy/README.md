# Mall Customer Segmentation using K-Means Clustering

## Abstract
This beginner-friendly Python project groups mall customers by annual income and spending score. It demonstrates data validation, exploratory data analysis (EDA), feature scaling, elbow analysis, K-Means training, visualization, and model persistence. A Streamlit dashboard supports CSV uploads, interactive charts, adjustable cluster counts, and result downloads.

## Problem statement
A mall may serve customers with different budgets and spending habits. Grouping similar customers can help analysts explore possible marketing strategies. This project discovers groups without predefined labels; it does not predict purchases or prove that a marketing campaign will succeed.

## Dataset and provenance
The bundled `dataset/Mall_Customers.csv` contains **500 synthetic customers**, generated with NumPy and seed 42. No original Mall Customers dataset was available in the workspace. The demo deliberately samples five income/spending groups, so clear separation is expected and is not evidence of real-world effectiveness. Age and gender are generated independently of those groups.

| Column | Meaning | Used for clustering? |
|---|---|---|
| CustomerID | Unique customer identifier | No |
| Gender | Descriptive category | No |
| Age | Age in years | No; EDA only |
| Annual Income (k$) | Annual income in thousands of dollars | Yes |
| Spending Score (1-100) | Spending indicator from 1 to 100 | Yes |

The program reports missing values before cleaning. Missing numeric measurements are replaced by each column's median; an entirely missing numeric column is rejected. Missing gender becomes `Unknown`. Invalid numbers, infinities, negative ages/incomes, out-of-range scores, missing/duplicate customer IDs, missing columns, and insufficient distinct observations produce readable errors. Rows are preserved rather than silently dropped. IDs and demographic categories are excluded from Euclidean distance.

## Project structure
```text
Mall_Customer_Segmentation/
├── dataset/
│   └── Mall_Customers.csv
├── main.py
├── app.py
├── requirements.txt
├── requirements-lock.txt       # Exact environment used for verification
├── README.md
├── VIVA_GUIDE.md
├── models/
│   └── kmeans_pipeline.pkl
└── outputs/
    ├── elbow_curve.png
    ├── customer_clusters.png
    ├── customer_eda.png
    ├── segmented_customers.csv
    ├── cluster_centers.csv
    ├── elbow_scores.csv
    └── metrics.json
```

## Algorithm explanation
1. Select annual income and spending score.
2. Standardize each feature by subtracting its mean and dividing by its standard deviation. This gives both features comparable influence on distance.
3. Fit K-Means for candidate values of k from 1 to at most 10. Inertia is the sum of squared distances to assigned centers, measured in standardized coordinates.
4. Plot inertia against k. The elbow is where adding more clusters starts giving smaller improvements. This implementation suggests the point with the greatest distance below the straight line joining the normalized endpoints. This heuristic is **not a guaranteed optimal k**; inspect the plot and compare nearby values.
5. Initialize k centers with K-Means++, assign each customer to the nearest center, then recompute centers as the average of their assigned customers. Repeat until convergence or the iteration limit.
6. Run ten initializations (`n_init=10`) and retain the fit with lowest inertia. Seed 42 makes results reproducible within the same software environment.
7. Inverse-transform centers into original income/spending units and save all assignments.

K-Means is useful here because it is fast, easy to explain, and provides interpretable centers for two numeric features. It works best with compact, roughly spherical groups. It is sensitive to outliers, feature scaling, initialization, and the chosen k. Cluster IDs are arbitrary labels, not rankings.

## Installation in VS Code
Install Python 3.11 or newer and the VS Code Python extension. Open this project folder using **File → Open Folder**, then open **Terminal → New Terminal**.

Windows PowerShell (activation is optional):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
In VS Code, run **Python: Select Interpreter** from the command palette and choose `.venv\Scripts\python.exe`.

macOS/Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
.venv/bin/python -m streamlit run app.py
```
Use `requirements-lock.txt` instead of `requirements.txt` to reproduce the exact tested dependency versions where supported. The normal requirements allow compatible updates.

## How to run
After choosing/activating your environment:
```bash
python main.py
python main.py --clusters 5
python main.py --data dataset/My_Customers.csv --clusters 4
python main.py --help
python -m streamlit run app.py
```
The default command automatically uses the elbow suggestion. Paths for bundled data and outputs are relative to the script, so launching from another working directory is supported. Custom relative paths are relative to your current terminal directory. Each CLI run replaces the saved model and output files. A missing default CSV is regenerated synthetically; a missing custom CSV produces an error.

The terminal displays schema information, missing values, summary statistics, selected k, every customer's assignment, cluster sizes, and center values. Open the PNG files in `outputs/` to view the saved charts.

Streamlit normally opens at `http://localhost:8501`. Use the bundled data or upload a CSV with the five required columns. Adjust k, hover over customers, zoom or pan, inspect centers and distributions, and download assignments. The dashboard does not overwrite the CLI model or bundled CSV. Stop the server with **Ctrl+C**.

## Results and interpretation
The included default run selected **4 clusters** and obtained a silhouette score of **0.637** on 500 customers. The simple elbow heuristic merges two of the five generated groups; this illustrates why an automatic suggestion requires inspection. Run `python main.py --clusters 5` or move the dashboard slider to compare five groups.

See `outputs/metrics.json` for the measured cluster count and silhouette score from the included run; `outputs/cluster_centers.csv` contains centers in original units. Silhouette ranges from -1 to 1, with higher values generally indicating stronger separation. It is not classification accuracy because no ground-truth customer labels are available.

At k=5, the deliberately generated groups approximately represent lower-income/lower-spending, lower-income/higher-spending, middle-income/middle-spending, higher-income/lower-spending, and higher-income/higher-spending customers. Match descriptions to center values rather than hard-coding cluster numbers. These descriptions are exploratory and should be validated with real customer research before making business decisions.

![Elbow curve](outputs/elbow_curve.png)
![Customer segments](outputs/customer_clusters.png)

## Saved model
`models/kmeans_pipeline.pkl` contains both the fitted scaler and K-Means estimator. Load only trusted pickle files. Use the same dependency versions when reloading.
```python
import pickle
import pandas as pd

with open("models/kmeans_pipeline.pkl", "rb") as file:
    model = pickle.load(file)
new_customers = pd.DataFrame({
    "Annual Income (k$)": [40, 90],
    "Spending Score (1-100)": [30, 80],
})
print(model.predict(new_customers))
```
New prediction features must be numeric and nonmissing. The saved pipeline handles scaling, while CSV validation and median imputation occur separately in `preprocess()` during analysis.

## Future improvements
- Validate on consented, representative real customer data.
- Compare k using silhouette analysis, stability across seeds, and business usefulness.
- Investigate outliers and compare DBSCAN or Gaussian mixture models.
- Add purchase frequency and recency with careful feature weighting.
- Fit and persist an imputer for a production prediction workflow.
- Monitor segment changes and evaluate campaigns with controlled experiments.

## Verification
Run `python -m unittest -v test_project.py` in the project environment. The checks cover customer preservation, missing-value imputation, invalid input rejection, original-unit centers, pipeline predictions, dashboard startup, and changing the dashboard cluster count.

## References
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Streamlit interactive Plotly charts](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart)

See [VIVA_GUIDE.md](VIVA_GUIDE.md) for a presentation plan and common questions.
