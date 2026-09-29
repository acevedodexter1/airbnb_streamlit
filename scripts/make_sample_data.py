"""Generates a small, deliberately messy Airbnb-style dataset (so the cleaning step has work to do)."""
import random, pandas as pd
random.seed(7)
PLACES = {  # city: (country, lat, lon, neighbourhoods, base price)
 "Manila": ("Philippines",14.60,120.98,["Makati","BGC","Ortigas","Poblacion"],70),
 "Cebu": ("Philippines",10.31,123.89,["IT Park","Mactan","Lahug"],55),
 "Siargao": ("Philippines",9.85,126.05,["General Luna","Cloud 9","Dapa"],60),
 "Baguio": ("Philippines",16.40,120.59,["Session Road","Camp John Hay"],45),
 "Tokyo": ("Japan",35.68,139.69,["Shibuya","Asakusa","Shinjuku"],130),
 "Bali": ("Indonesia",-8.65,115.22,["Ubud","Canggu","Seminyak"],85),
 "Bangkok": ("Thailand",13.75,100.50,["Sukhumvit","Silom","Ari"],65),
 "Paris": ("France",48.86,2.35,["Le Marais","Montmartre","Latin Quarter"],170)}
TYPES = {"Entire Home":1.5,"Private Room":0.7,"Studio":1.0,"Villa":2.4,"Shared Room":0.4}
ADJ = ["Cozy","Sunny","Modern","Seaside","Quiet","Charming","Luxury","Rustic","Bright","Minimalist"]
AMEN = ["Wifi","Kitchen","Pool","Air Conditioning","Parking","Washer","Beach Access","Workspace","Breakfast","Pet Friendly"]
POS = ["Amazing stay, super clean and the host was very friendly!","Loved the location, great value and comfortable beds.","Perfect place, beautiful view and spotless. Would book again!"]
NEU = ["It was okay, decent location but nothing special.","Average stay, basic amenities and fair price."]
NEG = ["Dirty bathroom and very noisy at night, disappointing.","Bad experience, host was slow to reply and the bed was uncomfortable."]
rows = []
for i in range(1, 421):
    city,(country,lat,lon,hoods,base) = random.choice(list(PLACES.items()))
    rt = random.choice(list(TYPES)); hood = random.choice(hoods)
    price = round(base*TYPES[rt]*random.uniform(.7,1.4))
    rating = round(min(5,max(3,random.gauss(4.5,.35))),2)
    txt = random.choice(POS if rating>4.4 else NEU if rating>3.9 else NEG)
    rows.append(dict(id=i,name=f"{random.choice(ADJ)} {rt} in {hood}",city=city,country=country,neighbourhood=hood,
      room_type=rt,price=f"${price:,.2f}",rating=rating,reviews=random.randint(0,320),
      bedrooms=1 if rt in("Studio","Shared Room","Private Room") else random.randint(1,4),
      guests=random.randint(1,8),amenities="|".join(random.sample(AMEN,random.randint(3,7))),
      description=f"A {rt.lower()} in {hood}, {city}. Great for travellers.",review_text=txt,
      latitude=lat+random.uniform(-.06,.06),longitude=lon+random.uniform(-.06,.06),
      last_review=f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"))
df = pd.DataFrame(rows)
# inject mess: missing values + duplicates
df.loc[df.sample(20,random_state=1).index,"rating"] = None
df.loc[df.sample(8,random_state=2).index,"price"] = None
df.loc[df.sample(10,random_state=3).index,"last_review"] = None
df = pd.concat([df, df.sample(12,random_state=4)])
df.to_csv("data/airbnb_listings.csv", index=False)
print("saved", len(df), "rows")
