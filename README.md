# TX Rural Broadband - Cell Tower Placement Optimizer

An interactive spatial analysis dashboard that maps broadband dead zones across Texas and uses a **Maximum Covering Location Problem (MCLP)** optimizer to identify where new cell towers would serve the most unserved residents.

---

## What It Does

- Maps all **existing cell towers** across Texas's 254 counties
- Highlights **dead zones** — areas with no coverage within a configurable radius
- Runs a **greedy MCLP optimizer** to recommend the top N tower locations by population impact
- Displays everything on an interactive **Leaflet map** with county-level broadband statistics and analytics charts

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/your-username/tx-rural-broadband.git
cd tx-rural-broadband

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python dashboard.py
```

Then open **http://localhost:8050** in your browser.

> **Note:** The app attempts to fetch live data from two government APIs on startup. If either is unavailable, it automatically falls back to synthetic data that mirrors the real schema.

---

## Data Sources

| Layer | Source |
|---|---|
| Cell towers | [HIFLD Open Data (DHS/CISA)](https://hifld-geoplatform.opendata.arcgis.com/datasets/cellular-towers) |
| County boundaries & population | [Census TIGER/Web](https://tigerweb.geo.census.gov/) |
| Broadband access % | FCC BDC *(synthetic placeholder — swap in real values)* |

All sources are U.S. government open data / public domain.

---

## How the Algorithm Works

**1. ETL** — Pull towers and county geometries from live APIs (with synthetic fallback).

**2. Find the Gaps** — Buffer each existing tower by the coverage radius, union all buffers into a "served" polygon, and flag everything outside as a dead zone.

**3. MCLP (greedy)** — Test 1,600 candidate locations across Texas. Pick the one covering the most unserved residents, mark them as served, then repeat for sites #2, #3, etc.

```
Maximise  Σ Pᵢ · yᵢ
where  yᵢ = 1 if resident i is within range of the new tower
```

The greedy approach gives a ~63% approximation guarantee for coverage maximisation.

---

## Interactive Controls

| Control | Effect |
|---|---|
| Coverage radius slider | How far (km) one tower is assumed to reach |
| Sites slider | Number of new towers to recommend (1–5) |
| ▶ Run Optimizer | Re-runs MCLP with current settings |

---

## Project Structure

```
tx-rural-broadband/
├── dashboard.py        # Full app — ETL, MCLP, Dash layout & callbacks
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Authors

Vedh Jaishankar · Sharvi Sriperambudur · Rishabh Jain

---

## License

MIT — see `LICENSE`.
