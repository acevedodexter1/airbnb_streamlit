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


def chat_answer(question: str, df_all, df_view, history: list) -> str:
    """Chatbot grounded in the dataset: we pass stats + the most relevant rows as context."""
    terms = [t for t in re.findall(r"[a-z]{3,}", question.lower())]
    rel = df_view[df_view["search_blob"].apply(lambda s: any(t in s for t in terms))]
    rel = (rel if len(rel) else df_view).sort_values(["rating", "reviews"], ascending=False).head(12)
    cols = ["name", "location", "room_type", "price", "rating", "guests", "amenities"]
    ctx = (f"Dataset: {len(df_all)} listings, cities: {', '.join(sorted(df_all.city.unique()))}. "
           f"Avg price ${df_all.price.mean():.0f}, min ${df_all.price.min():.0f}, max ${df_all.price.max():.0f}.\n"
           f"Relevant listings (from user's current filters):\n{rel[cols].to_csv(index=False)}")
    msgs = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    out = llm("You are a helpful Airbnb assistant. Answer ONLY from the data given; if it's not there, say so. Be concise.",
              f"{ctx}\nConversation:\n{msgs}\nQuestion: {question}", 400)
    if out:
        return out
    # offline fallback
    q = question.lower()
    if "cheap" in q:
        rel = df_view.nsmallest(5, "price")
    elif "best" in q or "top" in q:
        rel = df_view.sort_values(["rating", "reviews"], ascending=False).head(5)
    lines = [f"- **{r['name']}** ({r['location']}) - ${r['price']:.0f}/night, {r['rating']}★" for _, r in rel.head(5).iterrows()]
    return "(AI offline - showing matching listings)\n" + "\n".join(lines)
