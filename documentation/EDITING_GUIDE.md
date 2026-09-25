# Editing the submission package

All supplied personal details remain editable. No guide identity, signature or institutional approval has been invented.

## Personal and college details

Edit `documentation/submission_details.json` in VS Code. It contains the supplied name Nigilan, register number 145731105, Sathyabama university, BE CSE AI and academic year 2026-2027. The guide field is `[Guide name - not assigned]`.

The PDF cover, certificate and acknowledgement each contain six editable form fields. A PDF editor that supports AcroForms can change them directly. These are separate fields: update each page if editing the PDF by hand. Some browser PDF viewers do not save form changes reliably; reopen your saved copy to confirm them.

For a consistent update across all report pages and the presentation, change the JSON and rebuild both artifacts. PowerPoint title-slide text is also directly editable. Do not add a signature or approval unless actually supplied by the institution.

## Report

Edit `documentation/Project_Report_Source.md` to change the report content. Its 44 explicit page sections use `---PAGE---` separators. Headings, tables, screenshots and four vector workflow diagrams are rendered by `scripts/build_report.py`. The script refuses to silently clip an overflowing page.

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\Scripts\python.exe scripts/build_report.py
```

The script updates these identical copies:

- `AI_Customer_Segmentation_Project_Report.pdf`
- `documentation/AI_Customer_Segmentation_Project_Report.pdf`
- `documentation/Project_Report.pdf`

The certificate remains an unsigned template for authorized college review. The acknowledgement is an editable student draft.

## Presentation

Open `documentation/Presentation.pptx` in PowerPoint to edit its text, tables and architecture shapes. Screenshots and scientific figures are embedded images. To reproduce it, installed Microsoft PowerPoint on Windows is required:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_presentation.ps1
```

The script reads the details JSON and measured model comparison CSV. It writes the full-title file under both the project root and documentation, plus `documentation/Presentation.pptx`. Rebuilding replaces manual edits to the generated deck; put lasting template changes in the script. Re-copy all aliases after direct manual editing if you want every filename to match.

## Screenshots, viva and packaging

`scripts/capture_screenshots.py` uses the running dashboard with Microsoft Edge to generate eight actual screenshots and verify login/upload/logout workflows. It starts and stops its own temporary loopback server. Use it with the default local demo account; do not capture personal uploads for a public submission.

The canonical final viva file is `VIVA_GUIDE_FINAL.md`, mirrored as `documentation/Viva_Guide.md`. The older `VIVA_GUIDE.md` remains as supplementary historical documentation.

```powershell
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe scripts/package_project.py
```

Artifact checks render all report pages and validate PDF fields, slide structure and local documentation links. PowerPoint also exports slide PNGs during creation. Review visual layouts after content changes. The ZIP excludes environments, local databases, credentials, logs and render scratch files; fresh users run `main.py` to recreate the analytics database.
