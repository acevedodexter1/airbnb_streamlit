"""GenAI helpers: Hugging Face Inference API (DeepSeek) with an offline fallback."""
import json, os, re
import streamlit as st

MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")  # change if your model id differs
POS = {"amazing","loved","perfect","beautiful","clean","spotless","friendly","great","comfortable","value"}
NEG = {"dirty","noisy","bad","disappointing","slow","uncomfortable","terrible","worst","broken"}


def _token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")


def llm(system: str, user: str, max_tokens: int = 500):
    """Returns model text, or None if no token / API error (callers then use the fallback)."""
    token = _token()
    if not token:
        return None
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=MODEL, token=token)
        r = client.chat_completion(messages=[{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                                   max_tokens=max_tokens, temperature=0.3)
        return r.choices[0].message.content
    except Exception as e:
        st.session_state["llm_error"] = str(e)[:250]
        return None


def lexicon_label(text: str) -> str:
    """Cheap offline sentiment, used for the whole-dataset chart."""
    w = set(re.findall(r"[a-z]+", str(text).lower()))
    s = len(w & POS) - len(w & NEG)
    return "Positive" if s > 0 else "Negative" if s < 0 else "Neutral"


def analyze_listing(name: str, review: str, description: str) -> dict:
    """LLM sentiment + pros/cons for one listing."""
    out = llm("You analyse Airbnb reviews. Reply with ONLY JSON: "
              '{"sentiment":"Positive|Neutral|Negative","score":-1..1,"pros":[..],"cons":[..],"summary":"one sentence"}',
              f"Listing: {name}\nDescription: {description}\nReview: {review}")
    if out:
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return {**json.loads(m.group(0)), "source": MODEL}
        except Exception:
            pass
    lab = lexicon_label(review)
    return {"sentiment": lab, "score": {"Positive": .7, "Neutral": 0, "Negative": -.7}[lab],
            "pros": [], "cons": [], "summary": review, "source": "offline keyword fallback"}


# Words that describe *what the user wants*, not *which listing* - they must not be used as search keywords.
STOP = {"the", "and", "for", "with", "how", "much", "what", "which", "where", "who", "are", "any", "have", "has", "you",
        "can", "get", "give", "show", "list", "find", "tell", "about", "please", "that", "this", "there", "some", "from",
        "near", "place", "places", "stay", "stays", "listing", "listings", "price", "prices", "cost", "night", "per",
        "top", "best", "cheap", "cheapest", "cheaper", "lowest", "smallest", "least", "budget", "affordable", "most",
        "expensive", "priciest", "highest", "luxury", "costly", "good", "nice", "rated", "rating", "ratings", "guests",
        "guest", "people", "pax", "under", "below", "than", "less", "within", "around", "peso", "pesos", "recommend"}
CHEAP = {"cheap", "cheapest", "cheaper", "lowest", "smallest", "least", "budget", "affordable", "low"}
PRICEY = {"expensive", "priciest", "highest", "luxury", "costly", "premium"}


def _peso(x) -> str:
    return f"\u20b1{x:,.0f}"


def _retrieve(question: str, df_view):
    """Pick the listings that really answer the question (keywords, guests, budget) and sort them by intent."""
    q = question.lower()
    words = re.findall(r"[a-z]+", q)
    # 1) numeric conditions:  "for 6 guests", "under 3000"
    m = re.search(r"(\d+)\s*(?:guests?|people|persons?|pax)", q)
    if m:
        df_view = df_view[df_view.guests >= int(m.group(1))]
    m = re.search(r"(?:under|below|less than|within|max|budget of)\s*(?:\u20b1|php|p)?\s*([\d,]+)", q)
    if m:
        df_view = df_view[df_view.price <= float(m.group(1).replace(",", ""))]
    # 2) keywords (city, district, amenity, room type...): keep the rows that match the MOST keywords
    terms = [{t, t[:-1]} if t.endswith("s") and len(t) > 3 else {t} for t in words if len(t) >= 3 and t not in STOP]
    if terms and len(df_view):
        score = df_view["search_blob"].apply(lambda b: sum(any(v in b for v in vs) for vs in terms))
        df_view = df_view[score == score.max()] if score.max() > 0 else df_view.iloc[0:0]
    # 3) sort by what the user cares about
    if CHEAP & set(words):
        return df_view.sort_values(["price", "rating"], ascending=[True, False]), "cheapest first"
    if PRICEY & set(words):
        return df_view.sort_values(["price", "rating"], ascending=[False, False]), "most expensive first"
    return df_view.sort_values(["rating", "reviews"], ascending=False), "best rated first"


def chat_answer(question: str, df_all, df_view, history: list) -> str:
    """Chatbot grounded in the dataset: we pass real statistics + the listings that match the question."""
    rel, order = _retrieve(question, df_view)
    cols = ["name", "neighbourhood", "city", "room_type", "price", "rating", "guests", "amenities"]
    city_stats = (df_view.groupby("city").price.agg(["count", "min", "mean", "max"]).round(0).astype(int)
                  .rename(columns={"count": "listings", "min": "min_price", "mean": "avg_price", "max": "max_price"}))
    ctx = (f"Currency: Philippine pesos (PHP), prices are per night. Dataset: {len(df_all)} listings in the Philippines.\n"
           f"Listings available under the user's current filters: {len(df_view)}. "
           f"Overall price range: {_peso(df_view.price.min())} to {_peso(df_view.price.max())}, average {_peso(df_view.price.mean())}.\n"
           f"Price stats per city:\n{city_stats.to_csv()}\n"
           + (f"{len(rel)} listings match the question (showing the first {min(len(rel), 10)}, {order}):\n"
              f"{rel[cols].head(10).to_csv(index=False)}" if len(rel) else "NO listing matches the question's keywords."))
    msgs = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    out = llm("You are StayFinder AI, a friendly assistant for a Philippines stay-search app. Answer ONLY from the data given; "
              "if it is not there, say so. All prices are in Philippine pesos - always write them like \u20b12,500 (never $). "
              "The listing rows are already sorted as the question requires, so the first row is the answer to "
              "'cheapest', 'best' or 'most expensive'. Be concise and friendly.",
              f"{ctx}\nConversation:\n{msgs}\nQuestion: {question}", 400)
    if out:
        return out
    # offline fallback (no token / API error): same retrieval, plain list
    if rel.empty:
        return "(AI offline) I couldn't find a stay matching that. Try another place, room type or amenity."
    lines = [f"- **{r['name']}** ({r['neighbourhood']}, {r['city']}) - {_peso(r['price'])}/night, {r['rating']}\u2605" for _, r in rel.head(5).iterrows()]
    return f"(AI offline - showing matching listings, {order})\n" + "\n".join(lines)