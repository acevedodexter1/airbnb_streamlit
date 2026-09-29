import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path: str = "data/airbnb_listings.csv") -> pd.DataFrame:
    """Load the CSV and clean it."""
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="id")                                   # remove duplicate listings
    df["price"] = pd.to_numeric(                                           # "$1,234.00" -> 1234.0
        df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df = df.dropna(subset=["name", "price", "latitude", "longitude"])      # unusable rows
    df = df[df["price"] > 0]
    df["rating"] = df["rating"].fillna(df["rating"].median()).round(2)     # impute missing ratings
    df["reviews"] = df["reviews"].fillna(0).astype(int)
    df["bedrooms"] = df["bedrooms"].fillna(1).astype(int)
    df["guests"] = df["guests"].fillna(1).astype(int)
    for c in ["name", "city", "country", "neighbourhood", "room_type"]:
        df[c] = df[c].astype(str).str.strip()
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["amenities"] = df["amenities"].fillna("")
    df["description"] = df["description"].fillna("")
    df["review_text"] = df["review_text"].fillna("")
    df["location"] = df["neighbourhood"] + ", " + df["city"] + ", " + df["country"]
    df["search_blob"] = (df["name"] + " " + df["location"] + " " + df["room_type"] + " "
                         + df["amenities"] + " " + df["description"]).str.lower()
    return df.reset_index(drop=True)
