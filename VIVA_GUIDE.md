For the final 50-question guide and login demo, see [VIVA_GUIDE_FINAL.md](VIVA_GUIDE_FINAL.md).

# Final-project viva and demonstration guide

## Opening statement
“My project is an AI-based customer analytics system. It combines transaction-derived RFM with income, spending score and age, compares four unsupervised learning algorithms, and converts segment profiles into transparent marketing suggestions. It includes SQLite storage, a six-page Streamlit dashboard and explanations for new-customer assignments. The demo uses synthetic data, so I do not claim real-world marketing effectiveness.”

## Eight-minute presentation
1. **Problem and objectives (1 minute):** Explain why two customers with similar income can differ in purchase timing and loyalty.
2. **Dataset and RFM (1 minute):** Show 2,000 customers and 32,800 purchases. Define the fixed analysis date and verify count/sum/latest-date aggregation.
3. **Preprocessing (1 minute):** Explain validation, fitted median imputation, log1p for RFM and standardization. Explain why IDs/gender are not distance inputs.
4. **Models (1 minute):** Describe K-Means, Ward, DBSCAN and GMM. Show the elbow curve and explain its heuristic status.
5. **Comparison (1 minute):** Show three metrics plus coverage. K-Means is recommended; Ward has a better DBI; DBSCAN rejects 304 customers. Explain why all three facts matter.
6. **Prediction (1 minute):** Enter age 35, income 95, spending 85, recency 12, frequency 28 and monetary 5500. Explain the resulting category, recommendation and feature support. Use the actual output rather than promising a specific label.
7. **Marketing and storage (1 minute):** Inspect a segment profile and audience export. Show dataset/run history in SQLite. Explain that campaigns are not sent.
8. **Limitations and future work (1 minute):** Discuss synthetic data, feature correlation, model assumptions, stability, and real campaign evaluation.

## Questions and answers
**1. What makes this an AI project?** It learns groups without target labels using unsupervised ML. The marketing layer is a separate rule-based component, not an LLM or learned response predictor.

**2. What is RFM?** Recency is days since last purchase, frequency is purchase count and monetary value is total spending in a stated analysis window.

**3. Why fix the analysis date?** Recency changes with time. A fixed date makes results reproducible and prevents the same dataset changing silently each day.

**4. Why use a transaction ledger?** It lets us verify that frequency equals counted purchases, monetary equals summed amounts and recency matches the latest date.

**5. What happens when there are no purchases?** Frequency and monetary become zero, last-purchase date is missing, and recency is 366. This sentinel means inactive within the window, not exactly 366 days since a known purchase.

**6. Why log-transform RFM?** Purchase counts and amounts can be highly skewed. log1p compresses large values and handles zero. It changes the geometry intentionally, so it must also be used during prediction.

**7. Why standardize?** Euclidean-distance methods can otherwise be dominated by larger numerical scales. Equal standardized weighting is still a modeling choice.

**8. How does K-Means work?** Initialize centers, assign points to the nearest center, update each center to its group's mean, and repeat to minimize within-cluster squared distances.

**9. Why K-Means++ and ten starts?** They reduce sensitivity to poor initialization. They do not guarantee a global optimum.

**10. How does Ward clustering differ?** It builds a hierarchy by merging groups to limit variance increases; it does not update a fixed set of k centers in the same way as K-Means.

**11. What is DBSCAN noise?** Points that do not belong to a density-connected cluster. Label -1 is not a normal customer segment and receives a review recommendation.

**12. What do eps and min_samples mean?** eps is the neighborhood radius in the transformed feature space; min_samples controls how many neighbors support a dense core.

**13. What does GMM add?** Covariance-aware component shapes and posterior membership probabilities. These probabilities are model-based, not guaranteed or calibrated correctness.

**14. What does the elbow show?** Diminishing reductions in K-Means inertia as k increases. The program's suggestion can be overridden because some data has no clear elbow.

**15. What do the three metrics mean?** Silhouette measures own-group fit versus another group; Davies–Bouldin measures relative cluster similarity; Calinski–Harabasz compares between-group and within-group dispersion. Higher silhouette/CH and lower DBI are preferred.

**16. Why report DBSCAN coverage?** A model can look well separated after discarding many difficult points. Showing the assigned percentage and enforcing an 80% eligibility threshold makes that tradeoff explicit.

**17. How is the recommended model chosen?** Average the model's ranks on three metrics and add a coverage penalty. Lowest score wins, with defined tie-breakers. This is a policy, not a universal definition of best.

**18. Why does K-Means win if Ward has better DBI?** The selection considers all three metrics. K-Means leads silhouette and CH on this demo, while Ward leads DBI.

**19. Can hierarchical clustering predict a new point?** Not natively in this implementation. We use a clearly labeled nearest-training-centroid proxy, which does not reconstruct the hierarchy.

**20. Can DBSCAN predict a new point?** It has no native predict method here. We assign using the nearest fitted core sample within eps, otherwise -1. This extension can differ from refitting on all data.

**21. Are the explanations causal or SHAP?** No. K-Means uses exact feature-wise distance differences against the next center; GMM similarity is descriptive and its actual assignment uses covariance/weights. The UI states these scopes.

**22. How are marketing recommendations created?** Explicit income/spending profile rules with frozen cohort medians, plus recency/frequency rules. Customers inherit their segment's suggestion. We do not claim individual product relevance or measured conversion lift.

**23. What is stored in SQLite?** Customer snapshots, purchase records, analysis runs, per-model labels and segment recommendations. Composite identifiers and foreign keys preserve associations and history.

**24. Why exclude CustomerID and gender?** ID distance is meaningless. Gender is retained descriptively and not used in similarity or marketing rules. Age remains a model input and would need assessment for a real application.

**25. How are missing values handled?** Partial missing model inputs use fitted medians persisted with the model. Entirely missing model columns and invalid values are rejected; missing gender becomes Unknown.

**26. Is the score accuracy?** No. There are no target labels. Internal metrics assess geometry, not correct customer categories or business impact.

**27. Why no train/test accuracy split?** This project demonstrates exploratory clustering and internal validation. Future work should assess stability and out-of-sample behavior; inventing accuracy without true labels would be misleading.

**28. What are the main limitations?** Synthetic data, fixed settings, correlated features, sensitivity to scaling/outliers, proxy predictions for some models, and no validation of marketing outcomes.

**29. What did you test?** Transaction reconciliation, missing/invalid inputs, model reload consistency, prediction behavior, noise handling, ranking policy, SQLite integrity, classic compatibility, all dashboard pages and browser CSV upload.

**30. What is the next practical improvement?** Validate on representative real data, evaluate segment stability, and test campaigns with controlled experiments before claiming revenue improvement.

## Demo startup
```powershell
cd "D:\ML project\AI_Customer_Segmentation"
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open http://localhost:8501. For a presentation without a live browser, use the screenshots, PNG charts, comparison CSV and sample prediction JSON in `outputs/`.
