import base64
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_loader import load_data
from src.genai import analyze_listing, chat_answer, lexicon_label

st.set_page_config(page_title="Airbnb Finder - Airbnb Search", page_icon="🏡", layout="wide")

# ---- Color palette (one place to change the whole look) ----
INK, TEAL, TEAL_DARK, TEAL_LIGHT = "#1F2A37", "#0E7C86", "#0A5C64", "#E6F2F3"
CORAL, CORAL_DARK, SAND, LINE, MUTED = "#FF6B5B", "#F0503F", "#F6F4EF", "#E8E4DA", "#6B7280"
GOOD, OKAY, BAD = "#2E9E6B", "#F2B84B", "#E4572E"


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# Chat avatars (SVG): AI = teal house with a coral sparkle, user = coral person
AI_AV = _svg_uri(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{TEAL_DARK}"/></linearGradient></defs><circle cx="32" cy="32" r="32" fill="url(#g)"/>
<path d="M32 17 14 32h5v15h26V32h5z" fill="#fff"/><rect x="28" y="37" width="8" height="10" rx="1.5" fill="{TEAL}"/>
<path d="M48 9l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="{CORAL}"/></svg>""")
USER_AV = _svg_uri(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="32" fill="{CORAL}"/>
<circle cx="32" cy="25" r="10" fill="#fff"/><path d="M13 54c2-12 10-16 19-16s17 4 19 16z" fill="#fff"/></svg>""")

# Chat-window CSS lives in a normal (non f-) string; @NAME@ tokens are swapped for palette colors
CHAT_CSS = """
.chat-head{display:flex;align-items:center;gap:12px}
.chat-head img{width:46px;height:46px;border-radius:50%;box-shadow:0 0 0 3px @TEAL_LIGHT@,0 4px 10px rgba(14,124,134,.3)}
.chat-head b{display:block;font-size:1.05rem;color:@INK@;line-height:1.2}
.chat-head span{font-size:.78rem;color:@MUTED@;display:flex;align-items:center;gap:6px}
.chat-head i{width:8px;height:8px;border-radius:50%;background:@GOOD@;display:inline-block;animation:online 2s infinite}
@keyframes online{0%{box-shadow:0 0 0 0 rgba(46,158,107,.5)}70%{box-shadow:0 0 0 7px rgba(46,158,107,0)}100%{box-shadow:0 0 0 0 rgba(46,158,107,0)}}
[data-testid="stChatMessage"]{background:transparent;padding:.3rem 0;gap:.55rem;align-items:flex-end}
[data-testid="stChatMessage"] img{border-radius:50%;box-shadow:0 2px 6px rgba(31,42,55,.2)}
[data-testid="stChatMessageContent"]{background:#fff;border:1px solid @LINE@;border-radius:16px 16px 16px 4px;padding:.55rem .85rem;
  width:fit-content;max-width:84%;flex:0 1 auto;box-shadow:0 1px 4px rgba(31,42,55,.06)}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
[data-testid="stChatMessageContent"] p,[data-testid="stChatMessageContent"] li{font-size:.92rem;line-height:1.45;margin:0 0 .35rem}
[data-testid="stChatMessageContent"] p:last-child,[data-testid="stChatMessageContent"] ul:last-child{margin-bottom:0}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]){flex-direction:row-reverse}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"]{
  background:linear-gradient(120deg,@TEAL_DARK@,@TEAL@);border:none;border-radius:16px 16px 4px 16px}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] *{color:#fff}
/* typing indicator (three bouncing dots) */
.typing{display:inline-flex;gap:5px;align-items:center;height:22px}
.typing span{width:8px;height:8px;border-radius:50%;background:@TEAL@;opacity:.3;animation:bounce 1.2s infinite ease-in-out}
.typing span:nth-child(2){animation-delay:.18s}.typing span:nth-child(3){animation-delay:.36s}
@keyframes bounce{0%,70%,100%{opacity:.3;transform:translateY(0)}35%{opacity:1;transform:translateY(-5px)}}
/* remove Streamlit's "Press Enter to submit form" hint inside the chat box */
[data-testid="stPopoverBody"] [data-testid="InputInstructions"],[data-testid="stForm"] [data-testid="InputInstructions"]{display:none!important}
/* quick-question chips */
[data-testid="stPopoverBody"] .stButton>button{border-radius:999px;font-weight:500;font-size:.85rem;padding:.35rem .8rem}
"""
for _k, _v in dict(TEAL_LIGHT=TEAL_LIGHT, INK=INK, MUTED=MUTED, GOOD=GOOD, LINE=LINE, TEAL_DARK=TEAL_DARK, TEAL=TEAL).items():
    CHAT_CSS = CHAT_CSS.replace(f"@{_k}@", _v)

st.markdown(f"""
<style>
/* ================= Base ================= */
.stApp{{background:{SAND}}}
.block-container{{padding-top:4.5rem;padding-bottom:6rem}}
::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-thumb{{background:#c9c4b6;border-radius:8px}}
::-webkit-scrollbar-thumb:hover{{background:{TEAL}}}

/* ================= Hero & KPI ================= */
.hero{{background:linear-gradient(120deg,{TEAL_DARK},{TEAL});color:#fff;padding:26px 32px;border-radius:20px;
      margin-bottom:18px;box-shadow:0 8px 24px rgba(14,124,134,.25)}}
.hero h1{{margin:0;font-size:2.1rem;color:#fff}}.hero p{{margin:4px 0 0;opacity:.92}}
.kpi{{background:#fff;border-radius:14px;padding:14px;text-align:center;border:1px solid {LINE};color:{MUTED};
     font-size:.85rem;box-shadow:0 1px 4px rgba(31,42,55,.05);border-top:4px solid {TEAL}}}
.kpi b{{font-size:1.55rem;color:{TEAL};display:block}}

/* ================= Stay cards ================= */
.card{{background:#fff;border-radius:16px;padding:16px 18px;margin-bottom:14px;box-shadow:0 2px 10px rgba(31,42,55,.07);
      border:1px solid {LINE};border-top:4px solid {TEAL};transition:all .2s ease;height:226px;
      display:flex;flex-direction:column;overflow:hidden}}
.card:hover{{transform:translateY(-4px);box-shadow:0 10px 24px rgba(31,42,55,.15);border-top-color:{CORAL}}}
.card>*{{flex-shrink:0}}
.card .top{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}}
.card h4{{margin:0 0 4px;font-size:1.02rem;line-height:1.3;color:{INK};display:-webkit-box;-webkit-line-clamp:2;
         -webkit-box-orient:vertical;overflow:hidden}}
.loc{{color:{MUTED};font-size:.85rem}}.loc.one{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.badge{{display:inline-block;background:{TEAL_LIGHT};border-radius:20px;padding:2px 10px;margin:6px 4px 0 0;font-size:.75rem;color:{TEAL_DARK}}}
.card .foot{{margin-top:auto;padding-top:6px}}
.price{{font-size:1.2rem;font-weight:700;color:{INK}}}.rate{{font-weight:700;color:{CORAL};white-space:nowrap}}
.tip{{background:#fff;border-left:5px solid {CORAL};border-radius:10px;padding:12px 16px;margin:4px 0 16px;color:{INK};
     box-shadow:0 1px 4px rgba(31,42,55,.05)}}

/* ================= Sidebar ================= */
section[data-testid="stSidebar"]{{background:#fff;border-right:1px solid {LINE}}}
section[data-testid="stSidebar"] h2{{color:{TEAL_DARK}}}

/* ================= Buttons ================= */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{{
  border-radius:12px;font-weight:600;padding:.55rem 1.2rem;border:1.5px solid {TEAL};color:{TEAL};background:#fff;
  transition:all .18s ease}}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{{
  background:{TEAL_LIGHT};border-color:{TEAL_DARK};color:{TEAL_DARK};transform:translateY(-1px);
  box-shadow:0 4px 12px rgba(14,124,134,.2)}}
.stButton>button:active,.stDownloadButton>button:active{{transform:translateY(0)}}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primaryFormSubmit"]{{
  background:linear-gradient(120deg,{TEAL_DARK},{TEAL});color:#fff;border:none;box-shadow:0 4px 14px rgba(14,124,134,.35)}}
.stButton>button[kind="primary"]:hover{{background:linear-gradient(120deg,#084a50,{TEAL_DARK});color:#fff}}

/* ================= Inputs, selects, sliders ================= */
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="base-input"]{{border-radius:10px!important}}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{{
  border-color:{TEAL}!important;box-shadow:0 0 0 3px rgba(14,124,134,.15)!important}}
[data-baseweb="tag"]{{background:{TEAL}!important;border-radius:8px!important}}
.stNumberInput button{{border-radius:8px}}

/* ================= Tabs (pill style) ================= */
.stTabs [data-orientation="horizontal"]:not([role="tablist"]){{border-bottom:none!important;box-shadow:none!important}}
.stTabs [role="tablist"]{{gap:8px;background:#fff;padding:6px;border-radius:14px;border:1px solid {LINE};
  box-shadow:0 1px 4px rgba(31,42,55,.05);width:fit-content;max-width:100%}}
.stTabs [role="tab"]{{border-radius:10px;padding:8px 18px;height:auto;color:{MUTED};font-weight:600;transition:all .15s;border:none}}
.stTabs [role="tab"]:hover{{background:{TEAL_LIGHT};color:{TEAL_DARK}}}
.stTabs [role="tab"][aria-selected="true"]{{background:{TEAL}!important;color:#fff!important}}
.stTabs [role="tab"][aria-selected="true"] *{{color:#fff!important}}
.stTabs [role="tab"]::after,.stTabs [role="tablist"]::after{{display:none!important}}

/* ================= Tables / dataframes ================= */
[data-testid="stDataFrame"]{{border:1px solid {LINE};border-radius:14px;overflow:hidden;background:#fff;
  box-shadow:0 2px 10px rgba(31,42,55,.06)}}
[data-testid="stTable"] table,.stMarkdown table{{border-collapse:separate;border-spacing:0;width:100%;border:1px solid {LINE};
  border-radius:12px;overflow:hidden;background:#fff}}
.stMarkdown th{{background:{TEAL};color:#fff!important;padding:10px 12px!important;text-align:left}}
.stMarkdown td{{padding:9px 12px!important;border-top:1px solid {LINE}!important}}
.stMarkdown tr:nth-child(even) td{{background:#faf8f3}}
.stMarkdown tr:hover td{{background:{TEAL_LIGHT}}}

/* ================= Expanders, alerts, charts, map ================= */
[data-testid="stExpander"]{{background:#fff;border:1px solid {LINE}!important;border-radius:14px;
  box-shadow:0 1px 4px rgba(31,42,55,.05)}}
[data-testid="stExpander"] summary{{font-weight:600;color:{TEAL_DARK}}}
[data-testid="stAlert"]{{border-radius:12px}}
[data-testid="stPlotlyChart"]{{background:#fff;border:1px solid {LINE};border-radius:14px;padding:8px;
  box-shadow:0 1px 6px rgba(31,42,55,.05)}}
[data-testid="stDeckGlJsonChart"],[data-testid="stMap"]{{border-radius:14px;overflow:hidden}}
h2,h3{{color:{INK}}}

/* ================= Chat window ================= */
{CHAT_CSS}

/* ================= Floating AI chat bubble (bottom-right) ================= */
div[data-testid="stPopover"]{{position:fixed;bottom:24px;right:24px;z-index:9999;width:auto}}
div[data-testid="stPopover"] > div > button{{background:{CORAL};color:#fff;border:none;border-radius:999px;
  padding:12px 22px;font-weight:700;box-shadow:0 6px 20px rgba(255,107,91,.45);animation:pulse 2.6s infinite}}
div[data-testid="stPopover"] > div > button:hover{{background:{CORAL_DARK};color:#fff;transform:translateY(-2px)}}
div[data-testid="stPopoverBody"]{{width:min(410px,92vw);border-radius:18px;border:1px solid {LINE};
  box-shadow:0 14px 40px rgba(31,42,55,.25)}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(255,107,91,.5)}}70%{{box-shadow:0 0 0 14px rgba(255,107,91,0)}}
  100%{{box-shadow:0 0 0 0 rgba(255,107,91,0)}}}}
</style>""", unsafe_allow_html=True)

df = load_data()


@st.cache_data
def add_sentiment(d):
    d = d.copy(); d["sentiment"] = d["review_text"].map(lexicon_label); return d


df = add_sentiment(df)

st.markdown('<div class="hero"><h1>🏡 Airbnb Finder</h1><p>Find your perfect stay - search, compare, and ask our AI.</p></div>',
            unsafe_allow_html=True) 

# ---------------- Sidebar filters ----------------
sb = st.sidebar
sb.header("🔎 Search & Filters")
q = sb.text_input("Search (name, location, amenity…)", placeholder="e.g. pool siargao")
cities = sb.multiselect("City / Location", sorted(df.city.unique()))
rtypes = sb.multiselect("Room type", sorted(df.room_type.unique()))
lo, hi = int(df.price.min() // 50 * 50), int(-(-df.price.max() // 50) * 50)
price = sb.slider("Price per night (₱)", lo, hi, (lo, hi), step=50)
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
for col, (lab, val) in zip(k, [("Stays found", len(f)), ("Avg price / night", f"₱{f.price.mean():,.0f}" if len(f) else "-"),
                               ("Avg rating", f"{f.rating.mean():.2f}" if len(f) else "-"),
                               ("Cities", f.city.nunique())]):
    col.markdown(f'<div class="kpi"><b>{val}</b>{lab}</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["🏠 Stays", "📊 Quick Insights", "✨ Review Check"])

# ---------------- Stays ----------------
with tab1:
    if f.empty:
        st.warning("No stays match your filters. Try widening them.")
    else:
        c1, c2 = st.columns([3, 2])
        n = c1.select_slider("Show", [12, 24, 48], value=12)
        view = c2.radio("View as", ["Cards", "Table"], horizontal=True)
        if view == "Cards":
            cols = st.columns(3)
            for i, (_, r) in enumerate(f.head(n).iterrows()):
                cols[i % 3].markdown(
                    f'<div class="card"><div class="top"><h4>{r["name"]}</h4><span class="rate">★ {r.rating}</span></div>'
                    f'<div class="loc one">📍 {r.neighbourhood}, {r.city}</div>'
                    f'<div><span class="badge">{r.room_type}</span><span class="badge">{r.guests} guests</span>'
                    f'<span class="badge">{r.bedrooms} bd</span><span class="badge">{r.reviews} reviews</span></div>'
                    f'<div class="foot"><span class="price">₱{r.price:,.0f}</span> <span class="loc">/ night</span></div></div>',
                    unsafe_allow_html=True)
        else:
            tbl = f.head(n)[["name", "location", "room_type", "price", "rating", "reviews", "guests", "bedrooms"]]
            st.dataframe(tbl, hide_index=True, width="stretch", column_config={
                "name": "Stay", "location": "Location", "room_type": "Room type",
                "price": st.column_config.NumberColumn("Price / night", format="₱%,d"),
                "rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=5, format="★ %.2f"),
                "reviews": "Reviews", "guests": "Guests", "bedrooms": "Beds"})
            st.download_button("⬇️ Download results (CSV)", f.drop(columns=["search_blob"]).to_csv(index=False),
                               "stayfinder_results.csv", "text/csv")
        st.subheader("Map"); st.map(f, latitude="latitude", longitude="longitude", size=200, color=TEAL)

# ---------------- Quick Insights (simple, plain-language) ----------------
with tab2:
    if f.empty:
        st.info("Nothing to show yet - adjust your filters.")
    else:
        st.caption("Simple charts that update automatically when you change the filters on the left.")
        clean = dict(plot_bgcolor="#fff", paper_bgcolor="#fff", margin=dict(l=0, r=10, t=10, b=0))

        # 1) Cheapest city
        st.subheader("💰 Which place is cheapest?")
        by_city = f.groupby("city").price.mean().round(0).sort_values().reset_index()
        cheap, pricey = by_city.iloc[0], by_city.iloc[-1]
        st.markdown(f'<div class="tip">On average, <b>{cheap.city}</b> is the cheapest (₱{cheap.price:,.0f}/night)'
                    + (f' and <b>{pricey.city}</b> is the most expensive (₱{pricey.price:,.0f}/night).' if len(by_city) > 1 else '.')
                    + '</div>', unsafe_allow_html=True)
        fig = px.bar(by_city, x="price", y="city", orientation="h", text="price",
                     color_discrete_sequence=[TEAL], labels={"price": "Average price per night (₱)", "city": ""})
        fig.update_traces(texttemplate="₱%{text:,.0f}", textposition="outside")
        fig.update_layout(height=max(260, 42 * len(by_city)), yaxis=dict(categoryorder="total descending"), **{**clean, "margin": dict(l=0, r=70, t=10, b=0)})
        fig.update_xaxes(tickprefix="₱", tickformat=",")
        st.plotly_chart(fig, width="stretch")

        # 2) Price ranges
        st.subheader("🏷️ How much do most stays cost?")
        bands = pd.cut(f.price, [0, 2000, 4000, 8000, float("inf")], labels=["Under ₱2,000", "₱2,000 - ₱4,000", "₱4,000 - ₱8,000", "Over ₱8,000"])
        band_df = bands.value_counts().reindex(bands.cat.categories).reset_index()
        band_df.columns = ["Price range", "Stays"]
        top = band_df.loc[band_df.Stays.idxmax()]
        st.markdown(f'<div class="tip">Most stays ({int(top.Stays)} of {len(f)}) cost <b>{top["Price range"]}</b> per night.</div>',
                    unsafe_allow_html=True)
        fig = px.bar(band_df, x="Price range", y="Stays", text="Stays", color_discrete_sequence=[CORAL])
        fig.update_layout(height=300, xaxis_title="", **clean)
        st.plotly_chart(fig, width="stretch")

        # 3) Guest mood
        st.subheader("😊 What do guests say?")
        mood = f.sentiment.value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
        mood.columns = ["Mood", "Reviews"]
        pos = mood.loc[mood.Mood == "Positive", "Reviews"].iloc[0] / len(f) * 100
        st.markdown(f'<div class="tip"><b>{pos:.0f}%</b> of reviews are positive. '
                    '(We read each review and look for happy or unhappy words.)</div>', unsafe_allow_html=True)
        fig = px.pie(mood, names="Mood", values="Reviews", hole=.5, color="Mood",
                     color_discrete_map={"Positive": GOOD, "Neutral": OKAY, "Negative": BAD})
        fig.update_layout(height=320, paper_bgcolor="#fff", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

        # 4) Optional detailed charts (kept from the original version)
        with st.expander("📈 More charts (optional)"):
            st.markdown("**Is a higher price = a better rating?** Each dot is one stay. Hover to see its name.")
            fig = px.scatter(f, x="price", y="rating", color="room_type", hover_name="name",
                             color_discrete_sequence=[TEAL, CORAL, OKAY, GOOD, "#7C5CBF", "#B08968"],
                             labels={"price": "Price per night (₱)", "rating": "Rating", "room_type": "Room type"})
            fig.update_layout(height=340, **clean); fig.update_xaxes(tickprefix="₱", tickformat=",")
            st.plotly_chart(fig, width="stretch")
            st.markdown("**Price by room type.** The box shows where most prices fall; the line inside is the middle price.")
            fig = px.box(f, x="room_type", y="price", color="room_type",
                         color_discrete_sequence=[TEAL, CORAL, OKAY, GOOD, "#7C5CBF", "#B08968"],
                         labels={"price": "Price per night (₱)", "room_type": ""})
            fig.update_layout(height=340, showlegend=False, **clean); fig.update_yaxes(tickprefix="₱", tickformat=",")
            st.plotly_chart(fig, width="stretch")

# ---------------- Review Check (AI on one listing) ----------------
with tab3:
    st.caption("Pick a stay and let AI summarize what guests think. Uses Hugging Face DeepSeek if `HF_TOKEN` is set, otherwise a simple keyword check.")
    term = st.text_input("🔍 Search a stay", placeholder="e.g. bahay kubo, villa, pool siargao, Cebu…", key="review_search",
                         help="Searches ALL stays by name, place, room type or amenity (your sidebar filters are ignored while you search).")
    searching = bool(term.strip())
    pool = df if searching else f                       # while searching, look through every stay
    for t in term.lower().split():
        pool = pool[pool.search_blob.str.contains(re.escape(t))]
    if searching:
        pool = pool.sort_values(["rating", "reviews"], ascending=False)

    if pool.empty:
        st.info(f"No stay found for “{term.strip()}”. Try a place (Siargao), a type (Villa, Bahay Kubo) or an amenity (Pool)."
                if searching else "No listings to analyze.")
    else:
        if searching:
            st.caption(f"✅ {len(pool)} stay{'s' if len(pool) != 1 else ''} found for “{term.strip()}”"
                       + (" - showing the top 50." if len(pool) > 50 else "."))
        top = pool.head(50)
        pick = st.selectbox("Choose a stay", top["name"] + "  (#" + top["id"].astype(str) + ")")
        row = pool[pool.id == int(pick.split("#")[1].strip(")"))].iloc[0]
        st.write(f"**{row['name']}** - {row.location} - ₱{row.price:,.0f}/night")
        st.info(f"Guest review: {row.review_text}")
        if st.button("✨ Analyze with AI", type="primary"):
            with st.spinner("Thinking…"):
                res = analyze_listing(row["name"], row.review_text, row.description)
            st.success(f"Guest mood: **{res.get('sentiment')}** (score {res.get('score')})")
            st.write(res.get("summary", ""))
            x, y = st.columns(2)
            x.markdown("**👍 Good things**\n" + "\n".join(f"- {p}" for p in res.get("pros", [])))
            y.markdown("**👎 Not so good**\n" + "\n".join(f"- {p}" for p in res.get("cons", [])))
            st.caption(f"Source: {res.get('source')}")
            if st.session_state.get("llm_error"):
                st.warning(f"AI call failed, used fallback: {st.session_state['llm_error']}")

# ---------------- Floating AI chatbot (bottom-right, on every tab) ----------------
WELCOME = {"role": "assistant", "content": "Hi! 👋 I'm **StayFinder AI**. Ask me about prices, ratings, places or amenities "
                                           "across the Philippines."}
TYPING = '<div class="typing"><span></span><span></span><span></span></div>'
QUICK = {"💰 Cheapest stays": "What are the cheapest stays?", "⭐ Best rated": "Which stays have the best rating?",
         "🏖️ Beach stays": "Show me stays with beach access", "🏊 Villa with pool": "Best rated villa with a pool"}
st.session_state.setdefault("chat", [WELCOME])
avatar = lambda role: AI_AV if role == "assistant" else USER_AV

with st.popover("💬 Ask AI"):
    h1, h2 = st.columns([5, 2], vertical_alignment="center")
    h1.markdown(f'<div class="chat-head"><img src="{AI_AV}"><div><b>StayFinder AI</b>'
                f'<span><i></i>Online · searching {len(f)} stays</span></div></div>', unsafe_allow_html=True)
    if h2.button("🗑️ Clear", key="clear_chat", help="Start a new chat", width="stretch"):
        st.session_state.chat = [WELCOME]

    box = st.container(height=340)
    quick = None
    with box:
        for m in st.session_state.chat:
            st.chat_message(m["role"], avatar=avatar(m["role"])).markdown(m["content"])
        chips = st.empty()
        if len(st.session_state.chat) == 1:            # quick questions only on a fresh chat
            with chips.container():
                cc = st.columns(2)
                for j, (label, question) in enumerate(QUICK.items()):
                    if cc[j % 2].button(label, key=f"quick_{j}", width="stretch"):
                        quick = question

    with st.form("chat_form", clear_on_submit=True, border=False):
        c1, c2 = st.columns([5, 1])
        prompt = c1.text_input("Message", placeholder="Ask about stays…", label_visibility="collapsed")
        sent = c2.form_submit_button("➤", type="primary")

    ask = quick or (prompt.strip() if sent else "")
    if ask:
        chips.empty()
        with box:
            st.chat_message("user", avatar=USER_AV).markdown(ask)
            with st.chat_message("assistant", avatar=AI_AV):
                slot = st.empty()
                slot.markdown(TYPING, unsafe_allow_html=True)            # typing indicator while the AI thinks
                ans = chat_answer(ask, df, f if len(f) else df, st.session_state.chat)
                slot.markdown(ans)
        st.session_state.chat += [{"role": "user", "content": ask}, {"role": "assistant", "content": ans}]