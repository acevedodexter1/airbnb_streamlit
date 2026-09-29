import re
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_loader import load_data
from src.genai import analyze_listing, chat_answer, lexicon_label

st.set_page_config(page_title="StayFinder - Airbnb Search", page_icon="🏡", layout="wide")

st.markdown("""
<style>
.stApp{background:#fafafa}
.hero{background:linear-gradient(120deg,#ff385c,#bd1e59);color:#fff;padding:28px 32px;border-radius:18px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem;color:#fff}.hero p{margin:4px 0 0;opacity:.9}
.card{background:#fff;border-radius:16px;padding:16px 18px;margin-bottom:14px;box-shadow:0 2px 10px rgba(0,0,0,.07);
      border:1px solid #eee;transition:.2s;height:190px}
.card:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,.12)}
.card h4{margin:0 0 4px;font-size:1.02rem;color:#222}.loc{color:#717171;font-size:.85rem}
.badge{display:inline-block;background:#f7f7f7;border-radius:20px;padding:2px 10px;margin:6px 4px 0 0;font-size:.75rem;color:#444}
.price{font-size:1.15rem;font-weight:700;color:#222}.rate{float:right;font-weight:600;color:#ff385c}
.kpi{background:#fff;border-radius:14px;padding:14px;text-align:center;border:1px solid #eee}
.kpi b{font-size:1.5rem;color:#ff385c;display:block}
section[data-testid="stSidebar"]{background:#fff}
</style>""", unsafe_allow_html=True)

df = load_data()


@st.cache_data
def add_sentiment(d):
    d = d.copy(); d["sentiment"] = d["review_text"].map(lexicon_label); return d


df = add_sentiment(df)

st.markdown('<div class="hero"><h1>🏡 StayFinder</h1><p>Search stays by name, location, price and more - with AI insights.</p></div>',
            unsafe_allow_html=True)

# ---------------- Sidebar filters ----------------
sb = st.sidebar
sb.header("🔎 Search & Filters")
q = sb.text_input("Search (name, location, amenity…)", placeholder="e.g. pool siargao")
cities = sb.multiselect("City / Location", sorted(df.city.unique()))
rtypes = sb.multiselect("Room type", sorted(df.room_type.unique()))
lo, hi = int(df.price.min()), int(df.price.max())
price = sb.slider("Price per night ($)", lo, hi, (lo, hi))
min_rating = sb.slider("Minimum rating", 3.0, 5.0, 3.0, 0.1)
guests = sb.number_input("Guests", 1, 10, 1)
all_am = sorted({a for s in df.amenities for a in s.split("|") if a})
amen = sb.multiselect("Amenities", all_am)
d0, d1 = df.last_review.min().date(), df.last_review.max().date()
period = sb.date_input("Last review between", (d0, d1), min_value=d0, max_value=d1)
sort = sb.selectbox("Sort by", ["Rating ↓", "Price ↑", "Price ↓", "Most reviewed"])

f = df.copy()
for t in q.lower().split():
    f = f[f.search_blob.str.contains(re.escape(t))]
if cities: f = f[f.city.isin(cities)]
if rtypes: f = f[f.room_type.isin(rtypes)]
f = f[f.price.between(*price) & (f.rating >= min_rating) & (f.guests >= guests)]
for a in amen: f = f[f.amenities.str.contains(a, regex=False)]
if isinstance(period, tuple) and len(period) == 2:
    f = f[f.last_review.between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))]
f = f.sort_values(*{"Rating ↓": (["rating", "reviews"],), "Price ↑": ("price",),
                    "Price ↓": ("price",), "Most reviewed": ("reviews",)}[sort],
                  ascending=(sort == "Price ↑"))

k = st.columns(4)
for col, (lab, val) in zip(k, [("Listings", len(f)), ("Avg price", f"${f.price.mean():.0f}" if len(f) else "-"),
                               ("Avg rating", f"{f.rating.mean():.2f}" if len(f) else "-"),
                               ("Cities", f.city.nunique())]):
    col.markdown(f'<div class="kpi"><b>{val}</b>{lab}</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Listings", "📊 Analytics", "🤖 AI Insights", "💬 Chatbot"])

with tab1:
    if f.empty:
        st.warning("No stays match your filters. Try widening them.")
    else:
        n = st.select_slider("Show", [12, 24, 48], value=12)
        cols = st.columns(3)
        for i, (_, r) in enumerate(f.head(n).iterrows()):
            cols[i % 3].markdown(
                f'<div class="card"><span class="rate">★ {r.rating}</span><h4>{r["name"]}</h4>'
                f'<div class="loc">📍 {r.location}</div>'
                f'<span class="badge">{r.room_type}</span><span class="badge">{r.guests} guests</span>'
                f'<span class="badge">{r.bedrooms} bd</span><span class="badge">{r.reviews} reviews</span><br><br>'
                f'<span class="price">${r.price:,.0f}</span> <span class="loc">/ night</span></div>',
                unsafe_allow_html=True)
        st.subheader("Map"); st.map(f, latitude="latitude", longitude="longitude", size=200)

with tab2:
    if f.empty:
        st.info("Nothing to chart yet.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(px.bar(f.groupby("city").price.mean().round(0).reset_index(), x="city", y="price",
                              title="Average price by city", color="price",
                              color_continuous_scale="Reds"), use_container_width=True)
        b.plotly_chart(px.histogram(f, x="price", nbins=30, title="Price distribution",
                                    color_discrete_sequence=["#ff385c"]), use_container_width=True)
        c, d = st.columns(2)
        c.plotly_chart(px.scatter(f, x="price", y="rating", color="room_type", hover_name="name",
                                  title="Price vs rating"), use_container_width=True)
        d.plotly_chart(px.pie(f, names="sentiment", title="Review sentiment (keyword-based)", hole=.45,
                              color="sentiment", color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f1c40f",
                                                                     "Negative": "#e74c3c"}), use_container_width=True)
        st.plotly_chart(px.box(f, x="room_type", y="price", title="Price by room type",
                               color="room_type"), use_container_width=True)

with tab3:
    st.caption("Uses a Hugging Face DeepSeek model when `HF_TOKEN` is set; otherwise a keyword fallback.")
    if f.empty:
        st.info("No listings to analyze.")
    else:
        pick = st.selectbox("Choose a listing", f.head(50)["name"] + "  (#" + f.head(50)["id"].astype(str) + ")")
        row = f[f.id == int(pick.split("#")[1].strip(")"))].iloc[0]
        st.write(f"**{row['name']}** - {row.location} - ${row.price:,.0f}/night")
        st.info(f"Review: {row.review_text}")
        if st.button("✨ Analyze with AI"):
            with st.spinner("Thinking…"):
                res = analyze_listing(row["name"], row.review_text, row.description)
            st.success(f"Sentiment: **{res.get('sentiment')}** (score {res.get('score')})")
            st.write(res.get("summary", ""))
            x, y = st.columns(2)
            x.markdown("**👍 Pros**\n" + "\n".join(f"- {p}" for p in res.get("pros", [])))
            y.markdown("**👎 Cons**\n" + "\n".join(f"- {p}" for p in res.get("cons", [])))
            st.caption(f"Source: {res.get('source')}")
            if st.session_state.get("llm_error"):
                st.warning(f"AI call failed, used fallback: {st.session_state['llm_error']}")

with tab4:
    st.caption("Ask about the dataset, e.g. “cheapest place in Bali?” or “best rated villa with a pool”.")
    st.session_state.setdefault("chat", [])
    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])
    if prompt := st.chat_input("Ask about stays…"):
        st.chat_message("user").write(prompt)
        with st.spinner("…"):
            ans = chat_answer(prompt, df, f if len(f) else df, st.session_state.chat)
        st.chat_message("assistant").write(ans)
        st.session_state.chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]
