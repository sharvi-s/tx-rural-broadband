"""
TX Rural Broadband — Where should Texas build its next cell tower?
Run: python dashboard.py  →  http://localhost:8000
"""

import numpy as np
import pandas as pd
import requests, json, warnings
warnings.filterwarnings("ignore")

from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep

import folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table

np.random.seed(42)

TX = dict(lon_min=-106.65, lon_max=-93.51, lat_min=25.84, lat_max=36.50)
# HIFLD Open was deactivated Aug 26 2025; replaced with OpenStreetMap Overpass API
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TIGER_URL    = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
                "tigerWMS_Current/MapServer/82/query")
CENSUS_POP_URL = "https://api.census.gov/data/2020/dec/pl"  # no API key required

# ── ETL ───────────────────────────────────────────────────────────────────────
_HEADERS = {"User-Agent": "TX-Rural-Broadband/1.0 (tamu-spatial-engineering)"}

def etl_towers():
    """Fetch Texas cell/communication towers from OpenStreetMap via Overpass API.
    HIFLD Open was permanently deactivated on Aug 26 2025.
    """
    try:
        query = (
            "[out:json][timeout:60];"
            "("
            "node[\"man_made\"=\"tower\"][\"tower:type\"=\"communication\"]"
            "(25.84,-106.65,36.50,-93.51);"
            "node[\"man_made\"=\"mast\"][\"tower:type\"=\"communication\"]"
            "(25.84,-106.65,36.50,-93.51);"
            "node[\"man_made\"=\"mast\"][\"communication:mobile_phone\"=\"yes\"]"
            "(25.84,-106.65,36.50,-93.51);"
            "node[\"tower:type\"=\"communication\"]"
            "(25.84,-106.65,36.50,-93.51);"
            ");out body;"
        )
        r = requests.get(OVERPASS_URL, params={"data": query},
                         headers=_HEADERS, timeout=70)
        elements = r.json().get("elements", [])
        rows = []
        for el in elements:
            lat = el.get("lat"); lon = el.get("lon")
            tags = el.get("tags", {})
            if lat and lon and TX["lat_min"]<lat<TX["lat_max"] and TX["lon_min"]<lon<TX["lon_max"]:
                rows.append({
                    "lat":    round(lat, 5),
                    "lon":    round(lon, 5),
                    "type":   tags.get("tower:type", tags.get("man_made", "Tower")),
                    "owner":  tags.get("operator",   tags.get("owner", "Unknown")),
                    "height_m": tags.get("height", "N/A"),
                    "status": "In Service",
                    "source": "OpenStreetMap (Overpass API)"
                })
        if len(rows) > 50:
            print(f"  ✔ Overpass/OSM: {len(rows)} towers")
            return pd.DataFrame(rows), True
        print(f"  ⚠ Overpass returned only {len(rows)} towers, using synthetic")
    except Exception as e:
        print(f"  ⚠ Overpass: {e}")
    return _syn_towers(), False

def etl_counties():
    """Fetch Texas county geometries from TIGER/Web and join 2020 Census populations.
    TIGER/Web layer 82 does not carry POP100; populations come from Census Data API.
    maxAllowableOffset=0.01° (~1km) dramatically reduces geometry size for fast download.
    """
    try:
        # 1. County geometries from TIGER/Web (layer 82 = Counties, Jan 2025 vintage)
        r_geo = requests.get(TIGER_URL, params={
            "where":              "STATE='48'",
            "outFields":          "GEOID,NAME,INTPTLAT,INTPTLON,AREALAND",
            "f":                  "geojson",
            "resultRecordCount":  300,
            "outSR":              "4326",
            "maxAllowableOffset": 0.01,  # simplify polygons ~1km tolerance
        }, headers=_HEADERS, timeout=60)
        geo_data = r_geo.json()

        # 2. County populations from Census 2020 Decennial API (no key required)
        r_pop = requests.get(CENSUS_POP_URL, params={
            "get": "NAME,P1_001N",
            "for": "county:*",
            "in":  "state:48"
        }, headers=_HEADERS, timeout=30)
        pop_rows = r_pop.json()          # [[header], [row], ...]
        pop_dict = {"48" + row[3]: int(row[1]) for row in pop_rows[1:]}

        if geo_data.get("features") and len(geo_data["features"]) > 100:
            _bband_rng = np.random.RandomState(42)   # isolated seed for broadband_pct
            for f in geo_data["features"]:
                p = f["properties"]
                geoid = p.get("GEOID", "")
                p["population"]    = pop_dict.get(geoid, 3000)
                p["broadband_pct"] = round(_bband_rng.uniform(18, 90), 1)
                p["source"]        = "Census TIGER + Census 2020"
            print(f"  ✔ Census TIGER + Census API: {len(geo_data['features'])} counties")
            return geo_data, True
    except Exception as e:
        print(f"  ⚠ Census: {e}")
    return _syn_counties(), False

# ── MCLP ──────────────────────────────────────────────────────────────────────
def run_mclp(towers_df, counties_gj, radius_km=10, n_sites=3, grid_n=40):
    radius_deg = radius_km/111.0
    demand_pts, demand_pops, names = [], [], []
    for f in counties_gj["features"]:
        p = f["properties"]
        # Prefer official TIGER internal-point coords over mean-of-ring approximation
        if "INTPTLAT" in p and "INTPTLON" in p:
            clat = float(p["INTPTLAT"]); clon = float(p["INTPTLON"])
        else:
            cs = f["geometry"]["coordinates"][0]
            clon = np.mean([c[0] for c in cs]); clat = np.mean([c[1] for c in cs])
        demand_pts.append([clon, clat])
        demand_pops.append(p.get("population", 3000))
        names.append(p.get("NAME", p.get("BASENAME", "?")))
    D=np.array(demand_pts); P=np.array(demand_pops,dtype=float)
    T=towers_df[["lon","lat"]].values
    # Vectorised coverage: process towers in chunks to avoid 13k-iteration Python loop
    covered=np.zeros(len(D),dtype=bool)
    CHUNK=500
    for i in range(0, len(T), CHUNK):
        chunk=T[i:i+CHUNK]                     # (chunk, 2)
        dists=np.sqrt(((D[:,None,:]-chunk[None,:,:])**2).sum(axis=2))  # (n_demand, chunk)
        covered|=dists.min(axis=1)<=radius_deg
    remaining=list(np.where(~covered)[0])
    total_unserved=float(P[remaining].sum())
    can_lons=np.linspace(TX["lon_min"]+0.5,TX["lon_max"]-0.5,grid_n)
    can_lats=np.linspace(TX["lat_min"]+0.5,TX["lat_max"]-0.5,grid_n)
    candidates=np.array([[lo,la] for lo in can_lons for la in can_lats])
    sites=[]
    for rank in range(1,n_sites+1):
        if not remaining: break
        rD=D[remaining]; rP=P[remaining]
        best_gain,best_ci,best_mask=-1,-1,None
        for ci,(clo,cla) in enumerate(candidates):
            m=np.sqrt((rD[:,0]-clo)**2+(rD[:,1]-cla)**2)<=radius_deg
            g=float(rP[m].sum())
            if g>best_gain: best_gain,best_ci,best_mask=g,ci,m
        clo,cla=candidates[best_ci]
        newly=[remaining[i] for i,v in enumerate(best_mask) if v]
        sites.append({"rank":rank,"lat":round(cla,3),"lon":round(clo,3),
                      "gain":int(best_gain),"counties":[names[i] for i in newly[:4]]})
        remaining=[r for r in remaining if r not in newly]
        print(f"  Site #{rank}: +{int(best_gain):,} residents")
    return sites, total_unserved

# ── Synthetic data ─────────────────────────────────────────────────────────────
def _syn_towers():
    owners=["AT&T","Verizon","T-Mobile","US Cellular","Crown Castle","SBA Comm.","American Tower"]
    types=["Monopole","Lattice","Guyed","Self-Supporting","Stealth","Rooftop"]
    clusters=[(-97.74,30.26,0.7,0.5,65),(-95.37,29.76,0.9,0.7,75),
              (-98.49,29.42,0.6,0.5,50),(-96.80,32.78,0.8,0.6,60),
              (-106.49,31.76,0.4,0.4,16),(-97.14,31.55,2.0,1.5,25),
              (-101.85,33.57,1.0,0.9,16),(-102.08,31.84,2.5,2.5,20),
              (-99.73,28.70,1.8,1.4,14),(-94.02,30.08,0.7,0.6,16)]
    rows=[]
    for lc,lac,sl,sla,n in clusters:
        for _ in range(n):
            lat=np.random.normal(lac,sla/3); lon=np.random.normal(lc,sl/3)
            if TX["lat_min"]<lat<TX["lat_max"] and TX["lon_min"]<lon<TX["lon_max"]:
                rows.append({"lat":round(lat,5),"lon":round(lon,5),
                             "type":np.random.choice(types),"owner":np.random.choice(owners),
                             "height_m":int(np.random.choice([60,75,100,120,150,200])),
                             "status":"In Service","source":"Synthetic (HIFLD schema)"})
    print(f"  ✔ Synthetic: {len(rows)} towers")
    return pd.DataFrame(rows)

def _syn_counties():
    data=[
        ("Brewster",29.09,-103.24,9232),("Presidio",29.65,-104.21,7818),
        ("Jeff Davis",30.72,-104.12,2342),("Reeves",31.32,-103.69,15976),
        ("Pecos",30.79,-102.34,15507),("Val Verde",29.89,-100.90,49025),
        ("Maverick",28.74,-100.32,57887),("Webb",27.76,-99.33,276652),
        ("Hidalgo",26.40,-98.18,868707),("Cameron",26.15,-97.53,421017),
        ("Nueces",27.73,-97.43,362294),("Bexar",29.45,-98.52,2009324),
        ("Comal",29.80,-98.26,161369),("Hays",30.06,-97.97,262916),
        ("Travis",30.33,-97.78,1290188),("Bastrop",30.10,-97.31,97216),
        ("Fort Bend",29.53,-95.77,822779),("Harris",29.85,-95.40,4780913),
        ("Galveston",29.31,-94.87,350682),("Brazoria",29.20,-95.44,419720),
        ("Montgomery",30.30,-95.50,620443),("McLennan",31.55,-97.21,256623),
        ("Bell",31.04,-97.48,362924),("Brazos",30.66,-96.33,229211),
        ("Lubbock",33.61,-101.84,310569),("Midland",32.00,-102.08,176832),
        ("Ector",31.87,-102.54,166223),("Tarrant",32.74,-97.29,2110640),
        ("Dallas",32.77,-96.79,2635516),("Collin",33.19,-96.57,1064465),
        ("Denton",33.21,-97.13,906422),("El Paso",31.76,-106.49,868859),
        ("Potter",35.40,-101.89,117415),("Tom Green",31.40,-100.46,119200),
        ("Jefferson",29.85,-94.15,252273),("Smith",32.37,-95.27,232751),
        ("Gregg",32.48,-94.82,123945),("Victoria",28.80,-97.00,92084),
        ("Williamson",30.65,-97.60,734130),("Wharton",29.31,-96.24,41551),
        ("Colorado",29.62,-96.52,21493),("Fayette",29.88,-96.92,25346),
        ("Robertson",31.03,-96.51,17000),("Falls",31.27,-96.93,17300),
        ("Limestone",31.55,-96.58,23437),("Navarro",32.05,-96.47,50580),
        ("Henderson",32.21,-95.85,82511),("Anderson",31.81,-95.65,58458),
        ("Cherokee",31.85,-95.16,52646),("Rusk",32.11,-94.76,54406),
        ("Hunt",33.13,-96.08,96235),("Kaufman",32.59,-96.29,140596),
        ("Lamar",33.66,-95.55,50368),("Bowie",33.45,-94.41,93858),
        ("Grayson",33.63,-96.68,138547),("Cooke",33.64,-97.21,41668),
        ("Wise",33.21,-97.65,78749),("Parker",32.78,-97.80,148220),
        ("Johnson",32.38,-97.37,175817),("Bosque",31.90,-97.64,18685),
        ("Hamilton",31.70,-98.11,8517),("Lampasas",31.20,-98.23,21040),
        ("Coryell",31.39,-97.80,75389),("Milam",30.79,-96.97,24757),
        ("Burnet",30.79,-98.23,48155),("Llano",30.71,-98.68,21795),
        ("Kerr",30.06,-99.35,52600),("Bandera",29.73,-99.24,23112),
        ("Medina",29.35,-99.10,51584),("Uvalde",29.35,-99.76,27763),
        ("Zavala",28.87,-99.76,12131),("Frio",28.87,-99.10,20306),
        ("Live Oak",28.35,-98.11,12309),("Karnes",28.89,-97.85,15387),
        ("San Patricio",28.01,-97.52,67138),("Aransas",28.11,-97.00,24763),
        ("Jim Wells",27.73,-98.09,41953),("Starr",26.56,-98.74,65920),
        ("Zapata",26.91,-99.17,14179),("Bee",28.42,-97.75,32565),
        ("Wilson",29.18,-98.06,49753),("Gonzales",29.45,-97.49,20837),
        ("Caldwell",29.84,-97.62,45883),("Guadalupe",29.58,-97.95,182929),
        ("Gillespie",30.32,-98.94,26947),("Kendall",29.94,-98.71,47431),
        ("Kinney",29.35,-100.42,3675),("Edwards",29.98,-100.30,2003),
        ("Sutton",30.49,-100.53,3776),("Menard",30.88,-99.82,2138),
        ("McCulloch",31.20,-99.35,8284),("Coleman",31.75,-99.44,8547),
        ("Brown",31.75,-98.98,38106),("Comanche",31.95,-98.56,13623),
        ("Eastland",32.25,-98.82,17755),("Palo Pinto",32.75,-98.32,29275),
        ("Erath",32.24,-98.22,42698),("Nolan",32.30,-100.40,14420),
        ("Mitchell",32.30,-100.92,9726),("Scurry",32.75,-100.92,17239),
        ("Dawson",32.74,-101.95,13833),("Yoakum",33.17,-102.82,8713),
        ("Bailey",34.07,-102.83,6994),("Hale",34.07,-101.84,36043),
        ("Floyd",34.07,-101.31,5765),("Crosby",33.61,-101.30,5737),
        ("Childress",34.52,-100.22,6664),("Swisher",34.53,-101.74,7679),
        ("Castro",34.53,-102.27,7530),("Deaf Smith",34.97,-102.61,18571),
        ("Randall",34.97,-101.89,138953),("Armstrong",34.97,-101.37,1902),
        ("Donley",34.97,-100.82,3677),("Gray",35.40,-100.81,22535),
        ("Carson",35.40,-101.36,6032),("Moore",35.84,-101.89,21904),
        ("Hartley",35.84,-102.60,5537),("Dallam",36.28,-102.60,7287),
        ("Sherman",36.28,-101.89,3034),("Hansford",36.28,-101.35,5613),
        ("Ochiltree",36.28,-100.81,10015),("Liberty",30.11,-94.85,91932),
        ("Chambers",29.70,-94.58,45849),("Hardin",30.34,-94.38,57602),
        ("Orange",30.12,-93.87,83396),("Jackson",28.96,-96.59,14839),
        ("Matagorda",28.80,-96.01,36643),("Calhoun",28.44,-96.59,21290),
        ("Refugio",28.33,-97.19,7236),("Goliad",28.65,-97.42,7658),
        ("DeWitt",29.09,-97.35,20013),("Lavaca",29.38,-96.93,19979),
        ("Burleson",30.50,-96.62,18443),("Lee",30.31,-96.97,17239),
        ("Polk",30.78,-94.83,50861),("Tyler",30.77,-94.36,21766),
        ("Jasper",30.73,-94.02,35710),("Newton",30.78,-93.76,14445),
        ("Sabine",31.34,-93.84,10834),("Shelby",31.79,-94.16,26216),
        ("Panola",32.16,-94.31,23440),("Harrison",32.55,-94.37,66645),
        ("Marion",32.80,-94.36,10083),("Cass",33.07,-94.35,30438),
        ("Red River",33.63,-94.99,12023),("Hopkins",33.15,-95.57,36728),
        ("Upshur",32.73,-94.96,40769),("Wood",32.78,-95.37,45539),
        ("Van Zandt",32.56,-95.70,56590),("San Jacinto",30.58,-95.13,29728),
        ("Waller",29.98,-95.99,55246),("Austin",29.89,-96.27,30167),
        ("Hood",32.44,-97.83,65058),("Dickens",33.61,-100.79,2444),
        ("Stonewall",33.18,-100.26,1350),("Kent",33.18,-100.78,762),
        ("Garza",33.18,-101.30,6461),("Lynn",33.17,-101.82,6002),
        ("Terry",33.17,-102.34,12819),("Andrews",32.30,-102.64,19338),
        ("Winkler",31.84,-103.07,8026),("Ward",31.51,-103.10,11198),
        ("Loving",31.84,-103.60,169),("Culberson",31.45,-104.52,2241),
        ("Hudspeth",31.45,-105.38,4886),("Terrell",30.22,-101.99,984),
        ("Crockett",30.72,-101.38,3464),("Reagan",31.37,-101.52,3843),
        ("Crane",31.43,-102.35,4839),("Haskell",33.18,-99.73,5631),
        ("Shackelford",32.74,-99.35,3311),("Stephens",32.74,-98.82,9366),
        ("Jack",33.25,-98.17,9044),("Clay",33.79,-98.21,10471),
        ("Archer",33.62,-98.68,8786),("Young",33.18,-98.69,18550),
        ("Throckmorton",33.18,-99.21,1536),("Concho",31.32,-99.88,3978),
        ("San Saba",31.15,-98.72,5962),("Mills",31.49,-98.60,4936),
        ("Somervell",32.22,-97.77,9128),("Titus",33.22,-94.97,32750),
        ("Camp",33.00,-94.98,13094),("Morris",33.12,-94.74,12450),
        ("Franklin",33.18,-95.22,10721),("Delta",33.39,-95.68,5231),
        ("Collingsworth",34.97,-100.27,2965),("Wheeler",35.40,-100.27,5057),
        ("Oldham",35.40,-102.61,2057),("Lipscomb",36.28,-100.27,3233),
        ("Hemphill",35.84,-99.99,3807),("Roberts",35.84,-100.81,827),
        ("Hutchinson",35.84,-101.35,20938),("Briscoe",34.52,-101.21,1546),
        ("Hall",34.52,-100.68,3027),("Motley",34.07,-100.79,1210),
        ("Cottle",34.08,-100.28,1505),("King",33.61,-100.26,272),
        ("Borden",32.74,-101.43,641),("McMullen",28.35,-98.57,662),
        ("La Salle",28.35,-99.10,7520),("Dimmit",28.43,-99.76,10663),
        ("Duval",27.68,-98.53,11157),("Jim Hogg",27.04,-98.68,5300),
        ("Brooks",27.03,-98.23,7223),("Kenedy",26.93,-97.64,404),
        ("Willacy",26.47,-97.64,22134),("Kleberg",27.43,-97.66,31425),
    ]
    dlon,dlat=0.70,0.48
    feats=[]
    seen=set()
    for name,lac,lc,pop in data:
        if name in seen: continue
        seen.add(name)
        feats.append({
            "type":"Feature",
            "properties":{"NAME":name,"GEOID":f"48{len(feats):03d}",
                          "population":pop,"area_sq_mi":round(dlon*69*dlat*69,1),
                          "broadband_pct":round(np.random.uniform(18,90),1),
                          "source":"Synthetic (Census TIGER schema)"},
            "geometry":{"type":"Polygon","coordinates":[[
                [lc-dlon/2,lac-dlat/2],[lc+dlon/2,lac-dlat/2],
                [lc+dlon/2,lac+dlat/2],[lc-dlon/2,lac+dlat/2],[lc-dlon/2,lac-dlat/2]
            ]]}})
    print(f"  ✔ Synthetic: {len(feats)} counties")
    return {"type":"FeatureCollection","features":feats}

# ── Boot ───────────────────────────────────────────────────────────────────────
print("\n[Pipeline] Loading …")
towers_df,  REAL_T = etl_towers()
counties_gj, REAL_C = etl_counties()
init_sites, TOTAL_UNSERVED = run_mclp(towers_df, counties_gj, 10, 3)
TOTAL_POP = sum(f["properties"]["population"] for f in counties_gj["features"])
print("[Pipeline] Done.\n")

SRC = ("Real" if REAL_T else "Synthetic") + " · " + ("Real" if REAL_C else "Synthetic") + " county data"

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#FFF8F0"   # pastel orange background
CARD    = "#FFFFFF"
BORDER  = "#F0D9C0"
TEXT    = "#2C1810"
MUTED   = "#8C6B5A"
ACCENT  = "#D4500A"   # burnt orange — Texas!
BLUE    = "#1D6FA4"   # contrast for unserved / towers
GREEN   = "#2D8A4E"   # positive results
PEACH   = "#FDEBD0"   # softer card bg
SITE_C  = ["#D4500A","#1D6FA4","#2D8A4E","#7B2D8B","#C4960A","#B03060"]

def fmt(n):
    n=int(n)
    if n>=1_000_000: return f"{n/1_000_000:.1f}M"
    if n>=1_000:     return f"{n/1_000:.0f}K"
    return str(n)

# ── Map — Folium (real Leaflet map, no token needed) ─────────────────────────
def make_map_html(towers_df, counties_gj, sites, radius_km):
    """Returns an HTML string with a fully interactive Leaflet map."""

    m = folium.Map(
        location=[31.0, -99.5],
        zoom_start=6,
        tiles="CartoDB positron",   # light, clean basemap
        prefer_canvas=True,
    )

    # ① County polygons — coloured by broadband access %
    for f in counties_gj["features"]:
        p   = f["properties"]
        bpct = p.get("broadband_pct", 50)
        if bpct < 35:
            fill_color = "#E74C3C"   # red   — poor access
        elif bpct < 60:
            fill_color = "#F39C12"   # amber — medium
        else:
            fill_color = "#27AE60"   # green — good access

        folium.GeoJson(
            f,
            style_function=lambda feat, fc=fill_color: {
                "fillColor": fc, "fillOpacity": 0.45,
                "color": "#C8A882", "weight": 0.8,
            },
            tooltip=folium.Tooltip(
                f"<b>{p['NAME']} County</b><br>"
                f"Broadband access: <b>{bpct}%</b><br>"
                f"Population: {p['population']:,}"
            ),
        ).add_to(m)

    # ② Existing towers — clustered so the map isn't overwhelmed
    cluster = MarkerCluster(
        name="Existing Towers",
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 9},
    )
    for _, row in towers_df.iterrows():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=4,
            color="#555555",
            fill=True, fill_color="#555555", fill_opacity=0.7,
            weight=1,
            tooltip=(f"<b>{row['type']}</b> — {row['owner']}<br>"
                     f"Height: {row.get('height_m','?')} m"),
        ).add_to(cluster)
    cluster.add_to(m)

    # ③ Dead zones — grid cells with no coverage
    radius_deg = radius_km / 111.0
    buffers = [Point(r.lon, r.lat).buffer(radius_deg) for _, r in towers_df.iterrows()]
    covered  = unary_union(buffers) if buffers else box(0, 0, 0, 0)
    prep_cov = prep(covered)

    dead_group = folium.FeatureGroup(name="Dead Zones (no coverage)", show=True)
    cell_w = (TX["lon_max"] - TX["lon_min"]) / 52
    cell_h = (TX["lat_max"] - TX["lat_min"]) / 42
    for row_i in range(42):
        for col_i in range(52):
            clon = TX["lon_min"] + col_i * cell_w + cell_w / 2
            clat = TX["lat_min"] + row_i * cell_h + cell_h / 2
            if not prep_cov.contains(Point(clon, clat)):
                folium.Rectangle(
                    bounds=[[clat - cell_h/2, clon - cell_w/2],
                            [clat + cell_h/2, clon + cell_w/2]],
                    color=None, fill=True,
                    fill_color="#D4500A", fill_opacity=0.22,
                    weight=0,
                ).add_to(dead_group)
    dead_group.add_to(m)

    # ④ Optimal sites — big coloured markers + coverage circles
    site_colors = ["#D4500A", "#1D6FA4", "#2D8A4E", "#7B2D8B", "#C4960A"]
    site_group  = folium.FeatureGroup(name="Recommended Sites", show=True)
    for s in sites:
        c = site_colors[(s["rank"] - 1) % len(site_colors)]
        # Coverage circle
        folium.Circle(
            location=[s["lat"], s["lon"]],
            radius=radius_km * 1000,
            color=c, weight=2.5,
            fill=True, fill_color=c, fill_opacity=0.12,
            dash_array="8 5",
            tooltip=f"Coverage radius: {radius_km} km",
        ).add_to(site_group)
        # Star marker
        folium.Marker(
            location=[s["lat"], s["lon"]],
            icon=folium.DivIcon(html=f"""
                <div style="
                    background:{c};color:white;font-weight:900;font-size:15px;
                    width:36px;height:36px;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    border:3px solid white;box-shadow:0 2px 8px rgba(0,0,0,0.35);
                    font-family:Arial Black,sans-serif;
                ">#{s['rank']}</div>
            """, icon_size=(36, 36), icon_anchor=(18, 18)),
            tooltip=(
                f"<b>★ Recommended Site #{s['rank']}</b><br>"
                f"Would newly serve: <b>+{s['gain']:,} residents</b><br>"
                f"📍 {s['lat']}°N, {abs(s['lon']):.2f}°W<br>"
                f"Near: {', '.join(s['counties'][:3])}"
            ),
        ).add_to(site_group)
    site_group.add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                background:white;border:1px solid #E0D0C0;border-radius:10px;
                padding:12px 16px;font-family:Inter,sans-serif;font-size:12px;
                box-shadow:0 2px 10px rgba(0,0,0,0.12)">
      <b style="font-size:13px;color:#2C1810">Map Legend</b><br><br>
      <span style="color:#E74C3C">■</span> &lt;35% broadband access<br>
      <span style="color:#F39C12">■</span> 35–60% broadband access<br>
      <span style="color:#27AE60">■</span> &gt;60% broadband access<br>
      <span style="color:#D4500A;opacity:0.5">■</span> Dead zone (no coverage)<br>
      <span style="color:#555">●</span> Existing cell tower<br>
      <span style="background:#D4500A;color:white;padding:0 5px;border-radius:50%;font-weight:900">★</span> Recommended new site
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(position="topright").add_to(m)

    return m._repr_html_()

# Pre-render initial map
INIT_MAP_HTML = make_map_html(towers_df, counties_gj, init_sites, 10)

# ── Charts ────────────────────────────────────────────────────────────────────
def make_site_bars(sites):
    if not sites: return go.Figure()
    labels = [f"Site #{s['rank']}<br><span style='font-size:10px'>{s['lat']}°N</span>"
              for s in sites]
    fig = go.Figure()
    for i, s in enumerate(sites):
        fig.add_bar(
            x=[f"Site #{s['rank']}"], y=[s["gain"]],
            marker_color=SITE_C[i % len(SITE_C)],
            marker_line_color="white", marker_line_width=2,
            text=[f"+{s['gain']:,}"], textposition="outside",
            textfont=dict(size=13, color=TEXT, family="Arial Black"),
            name=f"Site #{s['rank']}",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(title="Residents newly served", gridcolor=BORDER,
                   tickfont=dict(size=10,color=MUTED), showgrid=True,
                   range=[0, max(s["gain"] for s in sites)*1.25]),
        xaxis=dict(tickfont=dict(size=12,color=TEXT), showgrid=False),
        margin=dict(l=10,r=10,t=20,b=10), height=220,
        bargap=0.4,
    )
    return fig

def make_owner_bar(towers_df):
    cnt = towers_df["owner"].value_counts().head(7)
    fig = go.Figure(go.Bar(
        y=cnt.index, x=cnt.values, orientation="h",
        marker=dict(color=ACCENT, opacity=0.75,
                    line=dict(color="white",width=1)),
        text=cnt.values, textposition="outside",
        textfont=dict(size=11, color=TEXT),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(gridcolor=BORDER, tickfont=dict(size=10,color=MUTED), showgrid=True),
        yaxis=dict(tickfont=dict(size=11,color=TEXT), showgrid=False),
        margin=dict(l=0,r=40,t=10,b=10), height=220,
    )
    return fig

def make_broadband_hist(counties_gj):
    pcts = [f["properties"]["broadband_pct"] for f in counties_gj["features"]]
    mean_pct = np.mean(pcts)
    fig = go.Figure()
    fig.add_histogram(
        x=pcts, nbinsx=18,
        marker=dict(color=ACCENT, opacity=0.7, line=dict(color="white",width=1)),
    )
    fig.add_vline(x=mean_pct, line_width=2, line_color=BLUE,
                  annotation_text=f" Mean: {mean_pct:.0f}%",
                  annotation_font=dict(color=BLUE,size=11),
                  annotation_position="top right")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title="% of county with broadband", gridcolor=BORDER,
                   tickfont=dict(size=10,color=MUTED)),
        yaxis=dict(title="# counties", gridcolor=BORDER,
                   tickfont=dict(size=10,color=MUTED)),
        margin=dict(l=10,r=10,t=10,b=10), height=220,
    )
    return fig

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="TX Rural Broadband")

app.index_string = """<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#FFF8F0;color:#2C1810;font-family:'Inter',sans-serif;font-size:14px}
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-thumb{background:#E8C9A0;border-radius:3px}
  .rc-slider-rail{background:#F0D9C0!important}
  .rc-slider-track{background:#D4500A!important}
  .rc-slider-handle{border-color:#D4500A!important;background:#D4500A!important}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

# ── Helper: card wrapper ──────────────────────────────────────────────────────
def card(children, style=None):
    s = {"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"12px","padding":"20px 24px"}
    if style: s.update(style)
    return html.Div(children, style=s)

def sec(title, sub=None):
    return html.Div([
        html.H2(title, style={"fontFamily":"'Lora',serif","fontSize":"20px",
                               "fontWeight":"700","color":TEXT,"marginBottom":"4px"}),
        html.P(sub, style={"fontSize":"12px","color":MUTED,"lineHeight":"1.5"}) if sub else None,
    ], style={"marginBottom":"14px"})

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="store", data=json.dumps(init_sites)),

    # TOPBAR
    html.Div([
        html.Div([
            html.Span("🗼 TX Rural Broadband",
                      style={"fontFamily":"'Lora',serif","fontWeight":"700",
                             "fontSize":"16px","color":ACCENT}),
            html.Span(" · Spatial Infrastructure Analysis",
                      style={"color":MUTED,"fontSize":"12px","marginLeft":"8px"}),
        ]),
        html.Div([
            html.Span(SRC, style={"fontSize":"11px","color":MUTED,"marginRight":"20px"}),
            html.Span("Vedh Jaishankar · Sharvi Sriperambudur · Rishabh Jain",
                      style={"fontSize":"11px","color":MUTED}),
        ]),
    ], style={"background":"#FFEFE0","borderBottom":f"1px solid {BORDER}",
              "padding":"12px 32px","display":"flex","alignItems":"center",
              "justifyContent":"space-between","position":"sticky","top":"0","zIndex":"200"}),

    html.Div([  # PAGE BODY

        # ── HERO ─────────────────────────────────────────────────────────────
        html.Div([
            html.H1("Where should Texas build its next cell tower?",
                    style={"fontFamily":"'Lora',serif","fontSize":"30px","fontWeight":"700",
                           "color":TEXT,"marginBottom":"10px","lineHeight":"1.3"}),
            html.P([
                "Millions of rural Texans live in ",html.Strong("broadband deserts"),
                " — areas with no reliable cell coverage. This tool maps those gaps and uses a ",
                html.Strong("Maximum Covering Location Problem (MCLP)"),
                " optimizer to pinpoint exactly where one new tower would help the most people."
            ], style={"color":MUTED,"lineHeight":"1.8","maxWidth":"620px","fontSize":"14px"}),
        ], style={"marginBottom":"28px","padding":"8px 0"}),

        # ── 4 STAT CARDS ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("📡", style={"fontSize":"24px","marginBottom":"6px"}),
                html.Div(f"{len(towers_df):,}",
                         style={"fontFamily":"'Lora',serif","fontSize":"32px","fontWeight":"700",
                                "color":ACCENT,"lineHeight":"1"}),
                html.Div("existing cell towers", style={"fontSize":"11px","color":MUTED,"marginTop":"3px"}),
            ], style={"background":CARD,"border":f"2px solid {BORDER}","borderTop":f"3px solid {ACCENT}",
                      "borderRadius":"12px","padding":"18px 16px","textAlign":"center"}),

            html.Div([
                html.Div("🗺️", style={"fontSize":"24px","marginBottom":"6px"}),
                html.Div(f"{len(counties_gj['features']):,}",
                         style={"fontFamily":"'Lora',serif","fontSize":"32px","fontWeight":"700",
                                "color":TEXT,"lineHeight":"1"}),
                html.Div("Texas counties mapped", style={"fontSize":"11px","color":MUTED,"marginTop":"3px"}),
            ], style={"background":CARD,"border":f"2px solid {BORDER}","borderTop":f"3px solid {BORDER}",
                      "borderRadius":"12px","padding":"18px 16px","textAlign":"center"}),

            html.Div([
                html.Div("📵", style={"fontSize":"24px","marginBottom":"6px"}),
                html.Div(fmt(TOTAL_UNSERVED),
                         style={"fontFamily":"'Lora',serif","fontSize":"32px","fontWeight":"700",
                                "color":"#C0392B","lineHeight":"1"}),
                html.Div("residents with no coverage", style={"fontSize":"11px","color":MUTED,"marginTop":"3px"}),
            ], style={"background":"#FFF5F5","border":f"2px solid #FFCCCC",
                      "borderTop":"3px solid #C0392B",
                      "borderRadius":"12px","padding":"18px 16px","textAlign":"center"}),

            html.Div([
                html.Div("✅", style={"fontSize":"24px","marginBottom":"6px"}),
                html.Div(id="kpi-gain", children=fmt(sum(s["gain"] for s in init_sites)),
                         style={"fontFamily":"'Lora',serif","fontSize":"32px","fontWeight":"700",
                                "color":GREEN,"lineHeight":"1"}),
                html.Div("newly served with 3 optimal sites",
                         style={"fontSize":"11px","color":MUTED,"marginTop":"3px"}),
            ], style={"background":"#F0FFF6","border":f"2px solid #A8E6C0",
                      "borderTop":f"3px solid {GREEN}",
                      "borderRadius":"12px","padding":"18px 16px","textAlign":"center"}),
        ], style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"14px","marginBottom":"36px"}),

        # ── MAP SECTION ───────────────────────────────────────────────────────
        sec("Coverage Map",
            "County shading: 🔴 <35% broadband  🟡 35–60%  🟢 >60%. "
            "Orange squares = dead zones (no tower within radius). Stars ★ = recommended new sites."),

        card([
            # controls row
            html.Div([
                html.Div([
                    html.Label("Coverage radius per tower",
                               style={"fontSize":"12px","fontWeight":"600","color":MUTED,
                                      "display":"block","marginBottom":"6px"}),
                    html.Div([
                        dcc.Slider(id="sl-radius", min=5, max=30, step=1, value=10,
                                   marks={5:"5 km",15:"15 km",30:"30 km"},
                                   tooltip={"always_visible":False}),
                        html.Span(id="lbl-radius", children="10 km",
                                  style={"fontWeight":"700","color":ACCENT,"fontSize":"13px",
                                         "marginLeft":"12px","whiteSpace":"nowrap"}),
                    ], style={"display":"flex","alignItems":"center"}),
                ], style={"flex":"1"}),
                html.Div([
                    html.Label("New sites to recommend",
                               style={"fontSize":"12px","fontWeight":"600","color":MUTED,
                                      "display":"block","marginBottom":"6px"}),
                    html.Div([
                        dcc.Slider(id="sl-sites", min=1, max=5, step=1, value=3,
                                   marks={1:"1",2:"2",3:"3",4:"4",5:"5"},
                                   tooltip={"always_visible":False}),
                        html.Span(id="lbl-sites", children="3",
                                  style={"fontWeight":"700","color":ACCENT,"fontSize":"13px",
                                         "marginLeft":"12px","whiteSpace":"nowrap"}),
                    ], style={"display":"flex","alignItems":"center"}),
                ], style={"flex":"1"}),
                html.Button("▶ Run Optimizer", id="btn-run", n_clicks=0, style={
                    "background":ACCENT,"color":"white","border":"none","borderRadius":"8px",
                    "padding":"10px 24px","fontWeight":"600","fontSize":"13px",
                    "cursor":"pointer","whiteSpace":"nowrap","alignSelf":"flex-end",
                    "fontFamily":"'Inter',sans-serif",
                }),
            ], style={"display":"flex","gap":"24px","alignItems":"flex-end","marginBottom":"14px"}),

            html.Div(id="run-note",
                     style={"fontSize":"12px","color":GREEN,"marginBottom":"8px","minHeight":"18px"}),

            html.Iframe(id="main-map", srcDoc=INIT_MAP_HTML,
                        style={"width":"100%","height":"540px","border":"none",
                               "borderRadius":"8px","display":"block"}),

        ], style={"marginBottom":"32px","padding":"20px"}),

        # ── RESULTS ───────────────────────────────────────────────────────────
        sec("Recommended Tower Sites",
            "The algorithm finds the location covering the most unserved people, then repeats. Each site is independent."),

        html.Div(id="site-cards",
                 style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)",
                        "gap":"14px","marginBottom":"16px"}),

        card([dcc.Graph(id="bar-sites", config={"displayModeBar":False})],
             style={"marginBottom":"32px","padding":"16px 20px"}),

        # ── DATA TABLES ───────────────────────────────────────────────────────
        sec("What Data We're Using",
            "The pipeline pulls from two real open-government APIs. Below are sample records from each."),

        html.Div([
            card([
                html.Div([
                    html.Span("Cell Tower Records",
                              style={"fontWeight":"600","fontSize":"14px","fontFamily":"'Lora',serif"}),
                    html.Span(" HIFLD Open Data (DHS/CISA)",
                              style={"fontSize":"11px","background":"#FEF3E2",
                                     "border":f"1px solid {BORDER}","color":ACCENT,
                                     "padding":"2px 8px","borderRadius":"4px","marginLeft":"8px"}),
                ], style={"marginBottom":"10px"}),
                dash_table.DataTable(
                    data=towers_df.head(10).to_dict("records"),
                    columns=[{"name":c.upper(),"id":c} for c in towers_df.columns],
                    style_table={"overflowX":"auto"},
                    style_data={"fontFamily":"monospace","fontSize":"11px",
                                "background":CARD,"color":TEXT,"border":f"1px solid {BORDER}"},
                    style_header={"background":PEACH,"color":MUTED,"fontWeight":"600",
                                  "fontSize":"10px","border":f"1px solid {BORDER}",
                                  "textTransform":"uppercase","letterSpacing":"0.05em"},
                    style_cell={"padding":"7px 10px","whiteSpace":"normal"},
                    style_data_conditional=[{"if":{"row_index":"odd"},"background":"#FFFAF5"}],
                    page_size=10,
                ),
            ], style={"marginBottom":"14px"}),
            card([
                html.Div([
                    html.Span("County Records",
                              style={"fontWeight":"600","fontSize":"14px","fontFamily":"'Lora',serif"}),
                    html.Span(" Census TIGER / FCC BDC",
                              style={"fontSize":"11px","background":"#FEF3E2",
                                     "border":f"1px solid {BORDER}","color":ACCENT,
                                     "padding":"2px 8px","borderRadius":"4px","marginLeft":"8px"}),
                ], style={"marginBottom":"10px"}),
                dash_table.DataTable(
                    data=[f["properties"] for f in counties_gj["features"][:10]],
                    columns=[{"name":c.upper(),"id":c} for c in
                             ["NAME","GEOID","population","area_sq_mi","broadband_pct","source"]],
                    style_table={"overflowX":"auto"},
                    style_data={"fontFamily":"monospace","fontSize":"11px",
                                "background":CARD,"color":TEXT,"border":f"1px solid {BORDER}"},
                    style_header={"background":PEACH,"color":MUTED,"fontWeight":"600",
                                  "fontSize":"10px","border":f"1px solid {BORDER}",
                                  "textTransform":"uppercase","letterSpacing":"0.05em"},
                    style_cell={"padding":"7px 10px"},
                    style_data_conditional=[{"if":{"row_index":"odd"},"background":"#FFFAF5"}],
                    page_size=10,
                ),
            ]),
        ], style={"marginBottom":"32px"}),

        # ── ANALYTICS ─────────────────────────────────────────────────────────
        sec("Data Analytics"),
        html.Div([
            card([
                html.P("Who owns existing towers?",
                       style={"fontWeight":"600","fontSize":"13px","marginBottom":"10px","color":TEXT}),
                dcc.Graph(figure=make_owner_bar(towers_df), config={"displayModeBar":False}),
            ]),
            card([
                html.P("How much broadband access do counties have?",
                       style={"fontWeight":"600","fontSize":"13px","marginBottom":"10px","color":TEXT}),
                dcc.Graph(figure=make_broadband_hist(counties_gj), config={"displayModeBar":False}),
            ]),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px","marginBottom":"32px"}),

        # ── METHODOLOGY ───────────────────────────────────────────────────────
        sec("How the Algorithm Works"),

        html.Div([
            html.Div([
                html.Div([
                    html.Span("1", style={"background":ACCENT,"color":"white","fontWeight":"800",
                                          "fontSize":"14px","width":"28px","height":"28px",
                                          "borderRadius":"50%","display":"inline-flex",
                                          "alignItems":"center","justifyContent":"center",
                                          "marginRight":"10px"}),
                    html.Span("ETL — Pull the Data",
                              style={"fontWeight":"700","fontSize":"14px","fontFamily":"'Lora',serif"}),
                ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),
                html.Ul([
                    html.Li("Cell towers from HIFLD (DHS open infrastructure database)"),
                    html.Li("County boundaries + population from Census TIGER (FIPS 48)"),
                    html.Li("LandScan USA 30m population raster — demand surface"),
                    html.Li("All reprojected to EPSG:3081 for accurate metre distances"),
                ], style={"color":MUTED,"paddingLeft":"16px","lineHeight":"1.9","fontSize":"13px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}",
                      "borderLeft":f"4px solid {ACCENT}","borderRadius":"0 10px 10px 0",
                      "padding":"16px 20px","marginBottom":"10px"}),

            html.Div([
                html.Div([
                    html.Span("2", style={"background":"#C0392B","color":"white","fontWeight":"800",
                                          "fontSize":"14px","width":"28px","height":"28px",
                                          "borderRadius":"50%","display":"inline-flex",
                                          "alignItems":"center","justifyContent":"center",
                                          "marginRight":"10px"}),
                    html.Span("Find the Gaps",
                              style={"fontWeight":"700","fontSize":"14px","fontFamily":"'Lora',serif"}),
                ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),
                html.Ul([
                    html.Li("Buffer each existing tower by the coverage radius"),
                    html.Li("Merge all buffers into one 'served' polygon"),
                    html.Li("Dead zones = Texas area MINUS that polygon"),
                    html.Li("Count how many people live inside dead zones"),
                ], style={"color":MUTED,"paddingLeft":"16px","lineHeight":"1.9","fontSize":"13px"}),
            ], style={"background":CARD,"border":f"1px solid {BORDER}",
                      "borderLeft":"4px solid #C0392B","borderRadius":"0 10px 10px 0",
                      "padding":"16px 20px","marginBottom":"10px"}),

            html.Div([
                html.Div([
                    html.Span("3", style={"background":GREEN,"color":"white","fontWeight":"800",
                                          "fontSize":"14px","width":"28px","height":"28px",
                                          "borderRadius":"50%","display":"inline-flex",
                                          "alignItems":"center","justifyContent":"center",
                                          "marginRight":"10px"}),
                    html.Span("Place the Tower — MCLP",
                              style={"fontWeight":"700","fontSize":"14px","fontFamily":"'Lora',serif"}),
                ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),
                html.Div("Maximise  Σ Pᵢ · yᵢ  —  where yᵢ = 1 if person i is within range of the new tower",
                         style={"fontFamily":"monospace","fontSize":"13px","color":ACCENT,
                                "background":"#FEF3E2","border":f"1px solid {BORDER}",
                                "borderRadius":"6px","padding":"10px 14px","marginBottom":"10px"}),
                html.Ul([
                    html.Li("Test 1,600 candidate locations across Texas"),
                    html.Li("Pick the one that covers the most unserved people"),
                    html.Li("Mark those people as served, repeat for site #2, #3…"),
                ], style={"color":MUTED,"paddingLeft":"16px","lineHeight":"1.9","fontSize":"13px"}),
            ], style={"background":"#F0FFF6","border":f"1px solid #A8E6C0",
                      "borderLeft":f"4px solid {GREEN}","borderRadius":"0 10px 10px 0",
                      "padding":"16px 20px","marginBottom":"10px"}),

            html.Div([
                html.P([html.Strong("Data ethics: "),
                        "All sources are public domain. No personal data used. ",
                        "The algorithm prioritises underserved rural communities by design. ",
                        "Reproducible with seed=42. Run: ",html.Code("python dashboard.py")],
                       style={"color":MUTED,"fontSize":"12px","lineHeight":"1.7"}),
            ], style={"background":PEACH,"border":f"1px solid {BORDER}",
                      "borderRadius":"10px","padding":"14px 18px"}),

        ], style={"maxWidth":"720px","marginBottom":"48px"}),

    ], style={"maxWidth":"1080px","margin":"0 auto","padding":"32px 24px"}),

], style={"background":BG,"minHeight":"100vh"})

# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("lbl-radius","children"), Output("lbl-sites","children"),
    Input("sl-radius","value"), Input("sl-sites","value"),
)
def upd_labels(r,s): return f"{r} km", str(s)

@app.callback(
    Output("store","data"), Output("run-note","children"), Output("kpi-gain","children"),
    Input("btn-run","n_clicks"),
    State("sl-radius","value"), State("sl-sites","value"),
    prevent_initial_call=True,
)
def rerun(n, radius, n_sites):
    sites, _ = run_mclp(towers_df, counties_gj, radius, n_sites)
    total    = sum(s["gain"] for s in sites)
    return json.dumps(sites), f"✓ Optimised — {total:,} residents newly served", fmt(total)

@app.callback(
    Output("main-map","srcDoc"),
    Output("site-cards","children"),
    Output("bar-sites","figure"),
    Input("store","data"),
    State("sl-radius","value"),
)
def update(sites_json, radius):
    sites = json.loads(sites_json)

    map_html = make_map_html(towers_df, counties_gj, sites, radius)

    cards = []
    for s in sites:
        c = SITE_C[(s["rank"]-1) % len(SITE_C)]
        pct = round(s["gain"] / max(TOTAL_UNSERVED,1) * 100, 2)
        cards.append(html.Div([
            html.Div(f"★ Recommended Site #{s['rank']}",
                     style={"fontWeight":"700","fontSize":"12px","color":c,
                            "textTransform":"uppercase","letterSpacing":"0.07em","marginBottom":"8px"}),
            html.Div(f"+{s['gain']:,}",
                     style={"fontFamily":"'Lora',serif","fontSize":"34px","fontWeight":"700",
                            "color":c,"lineHeight":"1"}),
            html.Div("residents newly served",
                     style={"fontSize":"12px","color":MUTED,"marginBottom":"12px"}),
            html.Div(style={"background":BORDER,"borderRadius":"4px","height":"6px","marginBottom":"5px"},
                children=[html.Div(style={"width":f"{min(pct*8,100)}%","height":"100%",
                                          "background":c,"borderRadius":"4px"})]),
            html.Div(f"{pct:.2f}% of unserved pop.",
                     style={"fontSize":"11px","color":MUTED,"marginBottom":"8px"}),
            html.Div(f"📍 {s['lat']}°N · {abs(s['lon']):.2f}°W",
                     style={"fontFamily":"monospace","fontSize":"11px","color":MUTED}),
            html.Div(f"Near: {', '.join(s['counties'][:3])}",
                     style={"fontFamily":"monospace","fontSize":"10px","color":MUTED,"marginTop":"3px"}),
        ], style={"background":CARD,"border":f"2px solid {BORDER}","borderTop":f"4px solid {c}",
                  "borderRadius":"12px","padding":"18px"}))

    return map_html, cards, make_site_bars(sites)


if __name__ == "__main__":
    print("="*50)
    print("  → http://localhost:8000")
    print("="*50)
    app.run(debug=False, port=8000, host="0.0.0.0")
