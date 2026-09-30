"""Generates a Philippines-only, deliberately messy Airbnb-style dataset (so the cleaning step has work to do).

REAL:   the destinations, districts/neighbourhoods and their map coordinates.
SAMPLE: listing names, prices, ratings and reviews are randomly generated (they are NOT actual Airbnb listings).
Run from the project root:  python scripts/make_sample_data.py
"""
import random, pandas as pd
random.seed(7)

# city: (region/province, weight, base price per night in Philippine pesos (PHP), coastal?, {district: (lat, lon)})
PLACES = {
 "Metro Manila": ("NCR", 6, 3000, False, {
     "Poblacion, Makati": (14.5656, 121.0300), "Bonifacio Global City": (14.5509, 121.0503),
     "Ortigas Center": (14.5866, 121.0567), "Malate": (14.5700, 120.9856),
     "Tomas Morato, Quezon City": (14.6280, 121.0340)}),
 "Cebu": ("Cebu", 5, 2500, True, {
     "Cebu IT Park": (10.3271, 123.9060), "Lahug": (10.3357, 123.8980), "Mactan Island": (10.3110, 123.9800)}),
 "Siargao": ("Surigao del Norte", 5, 2800, True, {
     "General Luna": (9.7869, 126.1590), "Cloud 9": (9.8091, 126.1669), "Dapa": (9.7590, 126.0570)}),
 "Baguio": ("Benguet", 4, 2100, False, {
     "Session Road": (16.4118, 120.5960), "Camp John Hay": (16.3970, 120.6120), "Burnham Park": (16.4120, 120.5930)}),
 "Boracay": ("Aklan", 5, 4700, True, {
     "Station 1": (11.9760, 121.9190), "Station 2": (11.9647, 121.9235), "Diniwid Beach": (11.9820, 121.9190)}),
 "Palawan": ("Palawan", 5, 3400, True, {
     "El Nido Town": (11.1784, 119.3930), "Coron Town": (12.0044, 120.2040), "Puerto Princesa City": (9.7392, 118.7353)}),
 "Bohol": ("Bohol", 3, 2800, True, {
     "Alona Beach, Panglao": (9.5490, 123.7750), "Tagbilaran City": (9.6470, 123.8540)}),
 "Davao": ("Davao del Sur", 3, 2000, False, {
     "Poblacion District": (7.0640, 125.6080), "Lanang": (7.1000, 125.6500), "Matina": (7.0500, 125.5900)}),
 "La Union": ("La Union", 3, 2300, True, {
     "Urbiztondo, San Juan": (16.6720, 120.3250)}),
 "Tagaytay": ("Cavite", 3, 3000, False, {
     "Tagaytay Ridge": (14.1153, 120.9621)}),
 "Vigan": ("Ilocos Sur", 2, 1800, False, {
     "Calle Crisologo": (17.5747, 120.3869)}),
 "Dumaguete": ("Negros Oriental", 2, 1800, True, {
     "Rizal Boulevard": (9.3068, 123.3054)}),
}
TYPES = {"Entire Home": 1.5, "Private Room": 0.7, "Studio": 1.0, "Villa": 2.4, "Shared Room": 0.4, "Bahay Kubo": 0.8}
TYPE_WEIGHTS = [3, 3, 3, 2, 2, 2]   # how often each type appears
ADJ = ["Cozy", "Sunny", "Modern", "Seaside", "Quiet", "Charming", "Luxury", "Rustic", "Bright", "Minimalist"]
AMEN = ["Wifi", "Kitchen", "Pool", "Air Conditioning", "Parking", "Washer", "Workspace", "Breakfast", "Pet Friendly"]
POS = ["Amazing stay, super clean and the host was very friendly!", "Loved the location, great value and comfortable beds.",
       "Perfect place, beautiful view and spotless. Would book again!"]
NEU = ["It was okay, decent location but nothing special.", "Average stay, basic amenities and fair price."]
NEG = ["Dirty bathroom and very noisy at night, disappointing.",
       "Bad experience, host was slow to reply and the bed was uncomfortable."]

cities, weights = list(PLACES), [PLACES[c][1] for c in PLACES]
rows = []
for i in range(1, 421):
    city = random.choices(cities, weights)[0]
    region, _, base, coastal, hoods = PLACES[city]
    hood = random.choice(list(hoods)); lat, lon = hoods[hood]
    rt = random.choices(list(TYPES), TYPE_WEIGHTS)[0]
    if rt == "Villa" and city in ("Metro Manila", "Baguio", "Vigan"):   # villas are rare in dense cities
        rt = "Entire Home"
    if rt == "Bahay Kubo" and city in ("Metro Manila", "Baguio"):       # nipa huts are for beach / countryside places
        rt = "Private Room"
    price = int(round(base * TYPES[rt] * random.uniform(.7, 1.4) / 50) * 50)   # nearest P50
    rating = round(min(5, max(3, random.gauss(4.5, .35))), 2)
    txt = random.choice(POS if rating > 4.4 else NEU if rating > 3.9 else NEG)
    amen = random.sample(AMEN, random.randint(3, 6)) + (["Beach Access"] if coastal and random.random() < .6 else [])
    rows.append(dict(
        id=i, name=f"{random.choice(ADJ)} {rt} in {hood.split(',')[0]}", city=city, country="Philippines",
        neighbourhood=hood, room_type=rt, price=f"₱{price:,.2f}", rating=rating, reviews=random.randint(0, 320),
        bedrooms=1 if rt in ("Studio", "Shared Room", "Private Room") else random.randint(1, 2) if rt == "Bahay Kubo" else random.randint(1, 4),
        guests=random.randint(1, 8), amenities="|".join(amen),
        description=f"A {rt.lower()} in {hood}, {city}, {region}. Great for travellers." + (" A traditional nipa hut." if rt == "Bahay Kubo" else ""), review_text=txt,
        latitude=lat + random.uniform(-.004, .004), longitude=lon + random.uniform(-.004, .004),   # ~400 m jitter
        last_review=f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"))
df = pd.DataFrame(rows)
# inject mess: missing values + duplicates
df.loc[df.sample(20, random_state=1).index, "rating"] = None
df.loc[df.sample(8, random_state=2).index, "price"] = None
df.loc[df.sample(10, random_state=3).index, "last_review"] = None
df = pd.concat([df, df.sample(12, random_state=4)])
df.to_csv("data/airbnb_listings.csv", index=False)
print("saved", len(df), "rows")