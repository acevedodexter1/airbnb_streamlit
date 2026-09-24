import os
import pandas as pd
import streamlit as st
import plotly.express as px
from huggingface_hub import InferenceClient

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Airbnb Review Sentiment Explorer", layout="wide")

client = InferenceClient(token=st.secrets["HF_TOKEN"])
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DATA_DIR = "data"
CACHE_FILE = f"{DATA_DIR}/reviews_with_sentiment.csv" 
SAMPLE_SIZE = 100  # how many reviews to send through the GenAI API


# ---------------------------------------------------------
# Load and clean the dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    listings = pd.read_csv(f"{DATA_DIR}/listings.csv")
    reviews = pd.read_csv(
        f"{DATA_DIR}/reviews.csv",
        engine="python",       # more tolerant of messy real-world text than the default engine
        on_bad_lines="skip",   # skip any row that doesn't match the expected column count
    )

    # Keep only the columns we need.
    # NOTE: column names can differ slightly between dataset versions —
    # check listings.columns / reviews.columns if this errors out.
    listings = listings[["id", "name", "neighbourhood", "room_type", "price"]]
    reviews = reviews[["listing_id", "date", "comments"]].rename(columns={"listing_id": "id"})

    reviews = reviews.dropna(subset=["comments"])
    reviews["comments"] = reviews["comments"].astype(str).str.strip()
    reviews = reviews[reviews["comments"].str.len() > 0]
    reviews = reviews.drop_duplicates(subset=["comments"])

    df = reviews.merge(listings, on="id", how="left")
    return df


# ---------------------------------------------------------
# GenAI sentiment analysis
# ---------------------------------------------------------
def get_sentiment_full(text):
    """Call the model and return the winning label, its confidence, and the
    full per-class score breakdown (used to show 'how the AI works')."""
    try:
        result = client.text_classification(text=text, model=SENTIMENT_MODEL)
        # result is a list of {"label": ..., "score": ...} for each class
        scores = {r["label"].capitalize(): r["score"] for r in result}
        top = max(result, key=lambda r: r["score"])
        return {
            "label": top["label"].capitalize(),  # -> "Positive" / "Negative" / "Neutral"
            "confidence": top["score"],
            "scores": scores,
        }
    except Exception as e:
        return {"label": f"Error: {e}", "confidence": 0.0, "scores": {}}


def get_sentiment(text):
    """Simple wrapper kept for convenience — just the winning label."""
    return get_sentiment_full(text)["label"]


@st.cache_data
def get_sample_with_sentiment(df, sample_size=SAMPLE_SIZE):
    # Cache results to a CSV so we don't re-call the API every time the app reloads.
    expected_rows = min(sample_size, len(df))
    if os.path.exists(CACHE_FILE):
        cached = pd.read_csv(CACHE_FILE)
        has_errors = cached.get("sentiment", pd.Series(dtype=str)).astype(str).str.startswith("Error").any()
        # Only trust the cache if it has no errors AND matches the current dataset size —
        # otherwise (e.g. more listings/reviews were added) it's stale and we recompute.
        if not has_errors and len(cached) == expected_rows:
            return cached

    sample = df.sample(expected_rows, random_state=42).copy()
    results = sample["comments"].apply(get_sentiment_full)
    sample["sentiment"] = results.apply(lambda r: r["label"])
    sample["confidence"] = results.apply(
        lambda r: round(r["confidence"] * 100, 1) if r["scores"] else None
    )
    sample.to_csv(CACHE_FILE, index=False)
    return sample


# ---------------------------------------------------------
# App layout
# ---------------------------------------------------------
df = load_data()
sample = get_sample_with_sentiment(df)

st.title("🏠 Airbnb Review Sentiment Explorer")
st.caption(
    f"Exploring {df['id'].nunique()} Airbnb listings and {len(df)} guest reviews, "
    "with sentiment scored by an AI language model."
)

with st.expander("🤖 How does the AI work?"):
    st.markdown(
        """
Every review below was read by an AI language model that decides whether it
sounds **Positive**, **Neutral**, or **Negative** — the same judgment a human
moderator would make, just automated and instant.

**Model:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
a RoBERTa transformer that was pretrained on a huge amount of text and then
fine-tuned specifically for 3-class sentiment classification.

**What happens to each review:**
1. The review text is sent to Hugging Face's hosted **Inference API** — no model runs on your machine.
2. The model scores the text against all three classes at once and returns a confidence for each (they add up to 100%).
3. The app picks the class with the highest confidence as the predicted **sentiment**.
4. Results are cached to `reviews_with_sentiment.csv` so the app doesn't re-call the API on every reload — the cache automatically refreshes if the dataset changes or a call previously failed.

Curious what the confidence scores actually look like? Scroll down to
**"Try it yourself"** and paste in your own review — it'll show the model's
score for all three classes, not just the winner.
        """
    )

# --- Sidebar filters ---
st.sidebar.header("Filters")
neighbourhoods = st.sidebar.multiselect(
    "Neighbourhood", sorted(sample["neighbourhood"].dropna().unique())
)
room_types = st.sidebar.multiselect(
    "Room type", sorted(sample["room_type"].dropna().unique())
)

filtered = sample.copy()
if neighbourhoods:
    filtered = filtered[filtered["neighbourhood"].isin(neighbourhoods)]
if room_types:
    filtered = filtered[filtered["room_type"].isin(room_types)]

# --- Table ---
st.subheader(f"Reviews ({len(filtered)})")
display_cols = ["name", "neighbourhood", "room_type", "price", "comments", "sentiment"]
if "confidence" in filtered.columns:
    display_cols.append("confidence")
st.dataframe(filtered[display_cols], use_container_width=True)

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    counts = filtered["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.bar(counts, x="sentiment", y="count", title="Sentiment Distribution", color="sentiment")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if not filtered.empty:
        by_neigh = (
            filtered.groupby("neighbourhood")["sentiment"]
            .apply(lambda s: (s == "Positive").mean())
            .reset_index(name="pct_positive")
        )
        fig2 = px.bar(
            by_neigh, x="neighbourhood", y="pct_positive",
            title="% Positive Reviews by Neighbourhood",
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- Try it yourself ---
st.subheader("Try it yourself")
st.caption("Paste any review to see exactly how the model scores it across all three classes.")
user_review = st.text_area("Paste an Airbnb review to analyze:")
if st.button("Analyze Sentiment"):
    if user_review.strip():
        result = get_sentiment_full(user_review)
        if result["scores"]:
            st.success(
                f"Predicted sentiment: **{result['label']}** "
                f"({result['confidence'] * 100:.1f}% confidence)"
            )
            scores_df = pd.DataFrame(
                {
                    "sentiment": list(result["scores"].keys()),
                    "confidence": [v * 100 for v in result["scores"].values()],
                }
            )
            fig3 = px.bar(
                scores_df, x="sentiment", y="confidence", color="sentiment",
                title="Model confidence by class", range_y=[0, 100],
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error(result["label"])
    else:
        st.warning("Please enter some text first.")