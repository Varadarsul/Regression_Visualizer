# Interactive Regression Explorer

This Streamlit app provides an interactive regression explorer using a predefined dataset (Diabetes from scikit-learn) and supports uploading your own CSV files.

Quick start

1. Create and activate a virtual environment (optional):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# or PowerShell: . .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Notes
- The app uses the sklearn Diabetes dataset by default.
- You can upload a CSV via the sidebar to analyze your own data (it will auto-detect numeric columns).
- For OLS step-by-step numeric, the app falls back to the pseudo-inverse if X^T X is singular.