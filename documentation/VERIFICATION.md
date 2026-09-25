# Final verification record

Verified locally on **25 September 2026**, using the project virtual environment on Windows. The data is synthetic. This record covers the exercised workflows and does not claim exhaustive correctness or production security certification.

| Check | Observed result |
|---|---|
| Automated suite | 11 tests passed (`python -m unittest -v test_project.py`) |
| Dependency consistency | `python -m pip check`: no broken requirements |
| Invalid login | Rejected in the real browser |
| Administrator login | `admin` / `admin123` opens the dashboard |
| Session controls | Forged token, revocation, idle expiry and account lockout checked by automated tests |
| Dashboard navigation | All six protected views exercised without application exceptions |
| Prediction | All four assignment methods exercised in Streamlit AppTest; browser prediction and explanation rendered |
| CSV upload | Actual upload widget accepted the bundled synthetic customer CSV and completed analysis |
| Logout and direct URL | Logout returns to login; a protected prediction URL remains gated |
| 3D visualization | WebGL chart rendered in Microsoft Edge |
| Responsive layout | Login and home rendered at 390 x 844; no document horizontal overflow |
| Screenshots | Eight actual desktop captures, mirrored in both requested folders |
| Dataset | 2,000 customer rows and 32,800 transaction rows; RFM aggregation covered by tests |
| SQLite | Round trip and foreign-key integrity covered by tests |
| Original features | Classic two-feature mode covered; historical source/data retained under legacy |
| PDF | 44 pages, 18 editable identity fields; every page rendered and visually reviewed |
| PowerPoint | 15 slides; native editable text, four tables and architecture shapes; reopened with PowerPoint and all slides reviewed |
| Documentation | Local Markdown targets checked; 50 numbered viva questions and answers included |

Browser evidence is recorded in [browser_verification.json](browser_verification.json). Artifact validation is reproducible with `scripts/verify_artifacts.py`; local render images and machine-readable checks are retained in `documentation/.qa/`, excluded from Git and the ZIP.

The PDF includes four vector diagrams: system architecture, ML workflow, data flow and K-Means workflow. Its cover, certificate and acknowledgement contain the user-supplied academic details as editable fields. The certificate remains unsigned and does not assert institutional approval. The full report source and presentation generator are supplied.

## Reproduce the checks

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_project.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\Scripts\python.exe scripts/capture_screenshots.py
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe scripts/package_project.py
```

The screenshot tool needs Microsoft Edge and the default local demo account. PPTX regeneration and native rendering require installed Microsoft PowerPoint. The application itself needs only `requirements.txt`.

## Scope and packaging

The portable source package includes synthetic datasets, fitted model bundles, generated outputs, screenshots and submission documents. It excludes `.venv`, Git metadata, local customer/authentication databases, credentials, logs and scratch renders. A fresh copy recreates the analytics database with `main.py` and creates the login database on first use. Saved models require a trusted source and compatible library versions.

Measurements are tied to the supplied seed, fixed reference date and configuration. The local shared administrator account, synthetic data and rule-based campaign suggestions remain documented limitations. No campaign effectiveness, real-customer accuracy or public-deployment security claim is made.
