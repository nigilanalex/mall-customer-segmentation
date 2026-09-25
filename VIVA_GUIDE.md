# College viva guide

## A five-minute presentation
1. **Problem (30 seconds):** Explain why customers with different income and spending patterns may need different approaches.
2. **Data (45 seconds):** Show the five columns and state clearly that the 500 records are synthetic. Explain that only income and spending score enter clustering; CustomerID is an identifier.
3. **Preparation (45 seconds):** Show missing-value counts, numeric validation, median imputation, distributions, and standardization.
4. **Algorithm (60 seconds):** Explain nearest-center assignment and mean-center updates. Show the elbow plot and explain why its recommendation needs judgment.
5. **Demo (60 seconds):** Run the dashboard, change k, hover over points, inspect centers, and download the assignments.
6. **Evaluation and limitations (60 seconds):** Report the measured silhouette score, explain arbitrary labels, disclose the deliberately separated synthetic groups, and suggest real-data validation.

## Common questions
**What is unsupervised learning?** Learning structure from input features without target labels.

**How does K-Means work?** Choose k centers, assign each point to its closest center, update each center to its group's mean, and repeat until stable. It minimizes within-cluster squared distances.

**Why K-Means?** It is efficient, understandable, and gives numerical centers that summarize customer groups. The chosen numeric features make its distance calculation easy to illustrate.

**Why scale?** Different units and spreads can dominate Euclidean distance. Standardization makes these two selected features comparable, though choosing equal influence remains a modeling assumption.

**Why exclude CustomerID?** Identifier differences do not express customer similarity. Age and gender are kept for description but excluded to focus the model on income/spending behavior.

**What is the elbow method?** Plot inertia versus k and look for diminishing improvements. Our automated knee detector is only a suggestion; some datasets have no clear elbow.

**Why not always choose the largest k?** Inertia decreases as k grows, even when extra clusters add little useful meaning. At one cluster per customer, inertia can be zero without useful segmentation.

**What is a centroid?** The mean feature vector of a cluster. We report centers after undoing standardization, so the units are interpretable.

**What does the silhouette score mean?** It compares how close a point is to its own group versus its nearest other group. Higher values generally suggest better separation; negative values suggest possible misassignment. It does not measure business value or classification accuracy.

**Why seed 42 and ten initializations?** The seed supports reproducibility. Multiple starts reduce the chance of choosing a poor local solution. The number 42 has no special mathematical benefit.

**Are cluster 0 and cluster 1 ordered?** No. Labels are arbitrary and may change between fits. Interpret centers instead.

**How are missing values handled?** Numeric measurements use column medians. Entirely missing numeric columns are rejected. Missing gender becomes Unknown; missing IDs are rejected.

**What are the limitations?** Sensitivity to outliers, k, and scaling; preference for compact groups; and no real-world evidence from synthetic data. The synthetic generator deliberately creates five groups.

**What would you improve?** Use representative real data, evaluate stability, add meaningful transaction features, compare clustering algorithms, and test business hypotheses independently.

## Suggested opening
“My project uses unsupervised learning to explore customer groups based on income and spending. I implemented validation, exploratory analysis, scaling, elbow selection, K-Means, and an interactive dashboard. The demonstration uses 500 synthetic customers, so I interpret the results as an algorithm demonstration rather than verified mall behavior.”
