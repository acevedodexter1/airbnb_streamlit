# 🏡 StayFinder - Philippines Stay Search (Streamlit + GenAI)

```
airbnb_app/
├── app.py                  # Streamlit UI (filters, stay cards, quick insights, AI review check, floating chatbot)
├── requirements.txt
├── data/airbnb_listings.csv
├── src/data_loader.py      # Pandas cleaning
├── src/genai.py            # Hugging Face DeepSeek + offline fallback
└── scripts/make_sample_data.py
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your HF_TOKEN
streamlit run app.py
```
All prices are in Philippine pesos (₱). Places are real Philippine destinations; listings/prices/reviews are generated sample data.
Optional: `HF_MODEL=<model-id>` overrides the default `deepseek-ai/DeepSeek-V4.1-Flash`.
No token? The app still works using the keyword-based fallback.

## Use your own dataset
Replace `data/airbnb_listings.csv` (keep the same column names) or edit `src/data_loader.py`.
Regenerate the sample: `python scripts/make_sample_data.py`.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (`.gitignore` already excludes secrets).
2. Go to share.streamlit.io → **New app** → pick repo, branch, main file `app.py`.
3. **Advanced settings → Secrets**: `HF_TOKEN = "hf_..."` → Deploy.

## Test & iterate checklist
- Search "pool", "villa boracay", "studio"; combine with price/rating/date filters.
- Try chatbot prompts: cheapest in Cebu, best rated in Siargao, "for 6 guests", a city that doesn't exist.
- Click the floating 💬 Ask AI button (bottom-right) - it stays visible on every tab.
- In Review Check, search "bahay kubo", "villa" or "pool siargao"; check it with and without a token (fallback path).