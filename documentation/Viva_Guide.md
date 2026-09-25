# Final Viva Guide

**AI-Based Customer Segmentation and Personalized Marketing Recommendation System using Machine Learning**

Prepare by understanding the data flow and opening the actual dashboard. Use the measured results below; do not describe silhouette as accuracy or synthetic segments as proven customer behavior.

## How to explain my project in 2 minutes

Good morning. My project is an AI-Based Customer Segmentation and Personalized Marketing Recommendation System using Machine Learning. Its aim is to help an analyst understand customer groups and choose suitable marketing actions.

The demonstration contains 2,000 synthetic customers and 32,800 transactions. I calculate RFM: recency is the number of days since the latest purchase, frequency is the purchase count, and monetary value is total spending in a defined observation window. I combine these with annual income, spending score and age.

The pipeline validates records, fills missing numerical values using fitted medians, log-transforms the RFM features and standardizes the selected inputs. I compare K-Means, Ward agglomerative clustering, DBSCAN and Gaussian Mixtures using silhouette, Davies-Bouldin and Calinski-Harabasz scores. The comparison also reports coverage because DBSCAN can leave customers unassigned.

On the included configuration, the recommendation policy selects K-Means with four clusters and a silhouette score of approximately 0.373. This is an internal separation measure, not 37.3 percent accuracy. The groups include premium, growth, value-focused and engagement profiles. A separate rule engine maps the group profiles to marketing suggestions.

The Streamlit dashboard has administrator login and six protected views. It supports CSV upload, interactive 2D and 3D charts, model comparison, new-customer assignment and explanations. SQLite stores datasets and analysis results, while saved model bundles preserve preprocessing for consistent prediction.

The main limitations are synthetic data, parameter sensitivity and recommendations that have not been tested in real campaigns. Future work would validate segments on representative data and measure campaign outcomes through controlled experiments.

## Fifty questions and answers

### 1. What is machine learning?

Machine learning builds models that learn patterns from data rather than encoding every decision manually. Here, clustering models learn groups; the marketing rules are separately written by the developer.

### 2. What is the difference between supervised and unsupervised learning?

Supervised learning uses labeled targets, such as a known churn outcome. Unsupervised learning explores structure without those targets. This project has no externally verified customer-segment labels, so it uses unsupervised clustering.

### 3. What is customer segmentation?

It is the process of grouping customers according to selected similarities. Here the similarities come from demographic and purchase features. A useful group should also be reviewed for business meaning, stability and potential action.

### 4. What problem does your project solve?

It combines customer data, transaction summaries, clustering comparisons and marketing suggestions in one inspectable workflow. It helps a student or analyst explore customer behavior without manually reviewing every record. It does not prove that a proposed campaign will succeed.

### 5. Why is the title AI-based?

The grouping stage uses machine learning, a branch of AI. The recommendation layer uses transparent rules applied to learned profiles. There is no LLM or generative AI API in this application.

### 6. Which dataset did you use?

I used reproducible synthetic data: 2,000 customers and 32,800 completed-purchase rows, generated with seed 42. The analysis date is 25 September 2026. The original 500-customer project is retained in the legacy folder.

### 7. Why use synthetic data, and what is its limitation?

It lets the project run without proprietary customer records and permits transaction/RFM consistency checks. However, the generator embeds behavioral patterns; success on these records does not establish effectiveness on real customers.

### 8. Which features enter advanced clustering?

Age, annual income, spending score, recency, purchase frequency and total purchase amount. CustomerID is only a join key. Gender and satisfaction are available for description but excluded from clustering distances and marketing rules.

### 9. Why exclude CustomerID and gender?

IDs identify records and do not represent meaningful numerical proximity. Gender is descriptive here and is not needed to define the chosen behavioral segments. Excluding a sensitive attribute alone does not guarantee fairness because other features can still be correlated with it.

### 10. What preprocessing is performed?

The program normalizes column aliases, validates IDs and values, computes or validates RFM, fills missing model values with fitted medians, log-transforms RFM and standardizes features. The fitted transformations are saved with the models.

### 11. How are missing values handled?

Partially missing numerical model features use the training cohort's median. Entirely missing required model features are rejected when no meaningful fitted median can be obtained. Missing transaction or customer identifiers are rejected rather than invented.

### 12. Why is feature scaling necessary?

Income, purchase totals, counts and days have different numerical ranges. Without scaling, large-magnitude units can dominate distance. Standard scaling uses z = (x - mean) / standard deviation, with training statistics reused at prediction time.

### 13. Why apply log1p to RFM?

RFM measurements can be highly skewed. log1p(x), or log(1 + x), compresses large values and handles zero. It changes the meaning of distance, so it is a modeling choice rather than a universally required preprocessing step.

### 14. What does RFM stand for?

Recency, frequency and monetary value. They summarize how recently a customer purchased, how often purchases occurred, and how much was spent during a stated observation window.

### 15. How do you calculate recency?

Subtract the latest qualifying purchase date from the fixed reference date and take the number of days. With latest purchase 20 September and reference 25 September, recency is five days. Lower recency means more recent activity.

### 16. How do you calculate frequency and monetary value?

Frequency counts completed purchase rows per customer within the window. Monetary value sums their amounts. For purchases of $120, $80 and $100, frequency is three and monetary value is $300.

### 17. What happens when a customer has no recent purchases?

The customer remains in the dataset. Frequency and monetary value become zero, the last qualifying date is missing, and recency is set to 366. This is a documented sentinel beyond the 365-day lookback, not a claim about the customer's actual last purchase date.

### 18. What is the observation window, and how do you reconcile RFM?

It runs from the reference date minus 365 days through the reference date, including both boundaries. Future purchases are rejected. Tests compare generated summaries with transaction counts, sums and latest dates. A supplied ledger overrides summary RFM values.

### 19. How does K-Means work?

Choose k, initialize k centers, assign each point to its nearest center, and update each center to the mean of its assigned points. Repeat until convergence or the iteration limit. The algorithm reduces within-cluster squared distances.

### 20. Why is K-Means used here?

It is efficient for the demonstration size, has understandable centers and supports native prediction for a new customer. It is also the original project's baseline. Comparing alternatives makes its assumptions and limitations visible.

### 21. What is k, and how do you choose it?

k is the requested number of clusters. The project fits K-Means over candidate values and uses a normalized-distance elbow heuristic. It suggests four on the bundled advanced dataset. The user can override this; the elbow is not proof of a unique optimal count.

### 22. What is inertia and the elbow method?

Inertia is the sum of squared distances from points to their assigned K-Means centers. Plot it against k and look for diminishing improvement. Inertia usually decreases as k increases, so simply choosing its minimum would encourage too many groups.

### 23. Why use K-Means++ and a random seed?

K-Means++ improves center initialization by spreading candidate centers. Multiple initializations reduce dependence on one starting configuration. Seed 42 makes the chosen random process reproducible in a compatible environment, but does not prove robustness across seeds.

### 24. What are K-Means limitations?

It is sensitive to scaling, outliers, initialization and the selected k. Squared Euclidean distance favors compact groups. Different shapes, varying densities or strong feature correlation may be represented poorly.

### 25. Are the displayed cluster centers identical to the model centers?

The dashboard reports original-unit group means for interpretation. The models operate after imputation, RFM log transformation and scaling. A mean on a transformed feature is not generally the transform of the original-unit mean, so those representations should not be confused.

### 26. What is hierarchical agglomerative clustering?

It begins with individual observations and merges groups successively. The project uses Ward linkage, which chooses merges based on the increase in within-group variance. The resulting hierarchy is cut at the selected cluster count.

### 27. How do you predict a new customer with Ward clustering?

The fitted estimator has no native predict method here. The project uses the nearest training-cluster centroid in the transformed space. This is labeled as a proxy and does not reproduce the entire hierarchy's merge process for the new customer.

### 28. How does DBSCAN work?

DBSCAN builds groups from dense neighborhoods. A core point has enough nearby observations within a radius, and connected dense regions form clusters. Points not attached to a cluster are labeled noise, represented by -1.

### 29. What are eps and min_samples?

eps is the neighborhood radius, and min_samples is the minimum support for a core point; scikit-learn includes the point itself. The bundled settings are eps 0.75 and min_samples 10. Radius is measured in transformed feature space, not dollars or days.

### 30. Why does DBSCAN produce noise, and how is a new customer assigned?

Sparse observations may not connect to a sufficiently dense region. The project assigns a new point to the nearest fitted core sample's group only if it lies within eps; otherwise it returns -1. This is an explicit extension, not native DBSCAN prediction or a refit.

### 31. What is a Gaussian Mixture Model?

A GMM models data as a weighted mixture of Gaussian distributions, estimating their means, covariances and weights. The project assigns each customer to the component with the highest posterior probability. Covariance lets it describe shapes beyond simple spherical groups.

### 32. How does GMM differ from K-Means?

K-Means assigns by nearest mean under squared Euclidean distance. GMM uses component distributions, covariance and weights and can provide soft membership probabilities. A posterior value should not automatically be described as calibrated confidence in a real customer category.

### 33. Why compare four algorithms?

They make different assumptions about geometry, density and uncertainty. Their comparison reveals whether the apparent grouping depends heavily on one method. It supports an informed choice rather than assuming that the original baseline must be best.

### 34. What is silhouette score?

For a point, let a be its mean distance to points in its own group and b its lowest mean distance to another group. Silhouette is (b - a) / max(a, b). Values range from -1 to 1; higher usually suggests better separation under the chosen distance.

### 35. What are Davies-Bouldin and Calinski-Harabasz scores?

Davies-Bouldin compares within-cluster scatter with separation; lower is preferred. Calinski-Harabasz compares between-group with within-group dispersion; higher is preferred. Both are internal geometric criteria and must be interpreted with feature choices and cluster assumptions.

### 36. Is a silhouette of 0.373 the same as 37.3% accuracy?

No. Accuracy compares predicted labels against known correct labels, which are unavailable here. Silhouette measures internal distance structure. The result suggests some separation with overlap; it does not quantify correct customer identities or marketing success.

### 37. How is the best model recommended?

Models need valid values for all three metrics and at least 80% coverage. The policy averages their metric ranks and adds a coverage penalty proportional to the unassigned fraction and number of eligible models. The lowest combined score wins, with deterministic tie-breaking. These are transparent policy choices.

### 38. Why report DBSCAN coverage and invalid partitions?

DBSCAN's internal metrics exclude noise, so it may be evaluated on an easier subset. Coverage reveals how many customers received groups. A single cluster, all noise or one group per point cannot provide a valid silhouette comparison, so these partitions do not receive a fabricated score.

### 39. What are the measured results?

K-Means produces four groups with silhouette 0.3730, Davies-Bouldin 1.0063 and Calinski-Harabasz 1330.10. Ward has silhouette 0.3689 and the best Davies-Bouldin score, 0.9797. DBSCAN creates three groups with 84.8% coverage and 304 noise customers. GMM creates four groups with silhouette 0.3549.

### 40. How are marketing recommendations generated?

The engine compares cluster mean income and spending with stored cohort median thresholds. High/high suggests premium actions; high/low suggests growth campaigns; low/high suggests value and loyalty offers; low/low suggests engagement. Average recency over 90 days adds win-back language, and high frequency adds loyalty benefits.

### 41. Are recommendations personalized to each individual?

They are personalized at the segment level. Customers inherit their assigned group's action, and unusual DBSCAN points receive review guidance. The system does not learn product preferences or individual campaign response and does not automatically contact customers.

### 42. How do you explain an assignment?

The output compares original customer values with group means and transformed feature gaps. K-Means uses feature-wise squared-distance support against the next center. Ward explains the proxy, DBSCAN its core-sample radius rule, and GMM its component assignment with descriptive similarities. None is a causal or SHAP explanation.

### 43. Why save preprocessing with the model?

New profiles must use the same feature order, imputation medians, log transforms and scaling learned during training. Refitting those on a single customer would change the feature space and break prediction consistency. The bundle also freezes profiles and recommendation thresholds.

### 44. What is Streamlit, and how do you run this application?

Streamlit creates interactive Python web applications. From the project root, run `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1`, open localhost:8501 and sign in. The six views cover home, analysis, clustering, comparison, prediction and recommendations.

### 45. How does authentication work?

The local admin account defaults to admin/admin123. Password verification uses salted PBKDF2-HMAC-SHA256; raw passwords are not stored. A random session token stays in server-side Streamlit session state, with its hash in SQLite. Five failures lock the account for one minute; logout revokes the token. Idle and absolute expiry are checked on interaction.

### 46. Why use session state and caching?

Streamlit reruns code after interactions. Session state retains the current authenticated user, selected analysis and form results for that session. Cached loading reduces repeated work for the bundled model. Expensive fitting is triggered through Run analysis instead of every navigation action.

### 47. Why use SQLite, and what does it store?

SQLite is a local relational database requiring no separate server. The analytics database stores immutable dataset snapshots, customers, transactions, runs, cluster results and recommendations with foreign-key relationships. A separate auth database stores users and session hashes. Local databases are excluded from public packaging.

### 48. How did you test the system?

Eleven automated tests cover RFM reconciliation, zero-purchase handling, invalid data, imputation, model round trips, assignment behavior, invalid metrics, coverage ranking, SQLite integrity, classic compatibility, dashboard navigation/predictions and authentication. Browser checks additionally exercise wrong/correct login, CSV upload, 3D rendering, logout and a protected URL. PDF and PPTX are opened, rendered and visually reviewed.

### 49. What are the main limitations and future improvements?

Synthetic data, correlated features, parameter sensitivity and unvalidated campaign rules limit practical conclusions. Internal metrics do not establish customer lifetime value or revenue impact. Next steps are representative real data, stability studies, product/refund modeling, controlled campaign experiments and managed identity for deployment.

### 50. What is your contribution, and how would you demonstrate it?

The contribution is an integrated educational workflow that retains the original K-Means analysis and adds auditable RFM, algorithm comparison, explicit prediction extensions, explainable segment actions, persistence and authenticated presentation. Demonstrate login, inspect the input and RFM, compare scores and coverage, predict a profile, show its explanation, export recommendations, then upload a sample and log out. Distinguish standard library algorithms from your integration and rule choices.

## Review-day demonstration order

1. Start the app before the review and keep the terminal running.
2. Show login, then identify the five dashboard KPIs.
3. Show customer and transaction schemas; explain the income-versus-monetary units.
4. Show the elbow and 2D/3D views; state that advanced clusters use six features.
5. Compare the metrics and point out DBSCAN's 304 noise customers.
6. Enter a customer, predict the segment and explain the method used.
7. Show the marketing rules and downloadable audience; do not claim a campaign was sent.
8. Upload the supplied synthetic CSV from the sidebar and run analysis.
9. Log out and finish with the limitations and future validation plan.

Keep the PDF and presentation available as offline backups. Numerical results are tied to the bundled seed/configuration and can change after retraining.
