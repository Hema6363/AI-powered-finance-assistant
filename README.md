# BudgetWise: AI-Powered Personal Finance Assistant

Analyze your expenses, visualize trends, and get personalized budget advice powered by Hugging Face.

## Features
- **Upload & Clean**: CSV/Excel ingestion with automatic column mapping (date, amount, category, merchant).
- **Analytics**: Monthly totals, top 3 categories, top 5 merchants (by frequency).
- **Visuals**: Matplotlib pie chart (category share) and bar chart (monthly spend).
- **AI Advice**: 3–5 tailored tips via Hugging Face Inference API (optional token). Fallback rule-based advice if no token.
- **Single Dashboard**: All insights presented in one Streamlit view.

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```

## Hugging Face Token (optional)
- Get a token at https://huggingface.co/settings/tokens (read access is enough).
- In the sidebar, paste your token into "Hugging Face API Token". Without a token, the app uses a built-in rule-based fallback.
- You can also set an environment variable before launching:
  - Windows PowerShell:
    ```powershell
    $env:HUGGINGFACEHUB_API_TOKEN = "YOUR_TOKEN"
    streamlit run app.py
    ```

## Data Format
The app tries to auto-detect common column names:
- **Amount**: `amount`, `amt`, `debit`, `money`, `value` (positive or negative supported)
- **Date**: `date`, `transaction date`, `posted date`, `booking date`, `time`
- **Category**: `category`, `cat`, `type`, `label`, `tag`
- **Merchant/Description**: `merchant`, `description`, `narration`, `details`, `payee`, `counterparty`

Notes:
- If amounts are negative for expenses, the app automatically flips signs to compute spending.
- Missing dates default to today (so monthly grouping is still shown).
- Missing category/merchant are labeled as `Uncategorized`/`Unknown`.

## Troubleshooting
- If file fails to parse, ensure it is `.csv`, `.xlsx`, or `.xls` and has an amount column.
- Large files: Streamlit runs in-memory; try reducing file size if needed.

## License
MIT
