# FloodGuard India

**Live AI flood detection and early warning for five of India's most flood-exposed river corridors.**

FloodGuard India pulls real-time river discharge and weather from Open-Meteo, turns the raw time series into a model-ready feature window, and runs a region-aware PyTorch BiLSTM to detect abnormal water levels and forecast the next eight days. A live FastAPI dashboard visualises every signal, and an email pipeline alerts subscribers when a basin reaches **Warning** or **Severe**.

> Experimental decision-support project. It is **not** an official flood warning service. Always follow IMD, CWC and local administration guidance.

---

## The five monitored basins

| Basin | River / station | Why it matters |
|---|---|---|
| Assam | Brahmaputra at Tezpur | Huge monsoon flows, bank erosion and embankment breaches |
| Bihar | Ganga at Patna | Low-lying middle Gangetic plain exposed to main-stem flooding |
| Kerala | Periyar at Aluva | Rapid monsoon bursts and reservoir releases upstream of Kochi |
| Mumbai | Ulhas at Kalyan | Extreme rainfall on impervious urban catchments |
| West Bengal | Hooghly at Kolkata | Compound upstream discharge, rain and tidal surge risk |

Coordinates sit on the GloFAS river network so the Flood API returns main-stem discharge rather than a dry grid cell.

---

## What the dashboard does

- **National flood gauge** — the highest active ensemble score across the five basins, with a plain-language summary.
- **Basin board** — observed discharge, 24-hour outlook, rainfall, a 24-day discharge sparkline, warning-threshold position and the eight-day peak for each basin.
- **Eight-day risk trajectory** — one line per basin with the Warning and Severe reference levels.
- **River watch** — per-basin detail with:
  - a 60-day rainfall + discharge hydrograph and the forecast tail drawn as a dashed segment,
  - a 60-day model risk timeline with shaded risk bands,
  - an eight-day forecast table,
  - the next-24-hour class probability distribution,
  - physical drivers (rainfall, discharge ratio, three-day trend, soil saturation).
- **Email alerts** — basin-specific subscriptions, test delivery, a clearly-labelled demo Warning, and the live dispatch log.
- **Model card** — architecture summary, training metadata, hold-out metrics, confusion matrix, loss curves and the 19 input features.
- **Operational behaviour** — automatic refresh every 30 minutes, live/cached/fallback status, and graceful retry when the API is unavailable.

The UI is dependency-light: vanilla JavaScript, hand-written CSS and a vendored Chart.js build. It is responsive down to phone width.

---

## How it works

```
Open-Meteo GloFAS  ─┐
Open-Meteo weather ─┼─► feature engineering ─► 14-day × 19-feature window
ERA5-Land history  ─┘                              │
                                                   ▼
                               region embedding ─► BiLSTM 2×64 ─► attention
                                                   │
                                                   ▼
                                    4-class probabilities (Normal/Watch/
                                    Warning/Severe) + 8-day risk path
                                                   │
                        40% model + 60% GloFAS discharge prior
                                                   │
                                                   ▼
                              hybrid score 0–100 → dashboard + email alerts
```

### Data sources

| Source | Used for |
|---|---|
| [Open-Meteo Flood API](https://open-meteo.com/en/docs/flood-api) (GloFAS) | Historical and live river discharge, plus the future GloFAS signal used by the hydrological prior |
| [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | Rainfall, soil moisture, temperature, wind, evapotranspiration |
| [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5-Land) | Training history from 2019 |

The live fetch has a short weather history window. The trainer uses the archive for long history, while operational inference uses the most recent live window and GloFAS forecast. If a network fetch fails, the last good local CSV is used and the basin is marked `cached`; if no cache exists, a deterministic synthetic fallback is marked `demo`.

### Risk labels and score

Risk labels come from basin-specific historical discharge quantiles:

| Label | Discharge rule |
|---|---|
| Normal | below the 80th percentile |
| Watch | 80th–92nd percentile |
| Warning | 92nd–97th percentile |
| Severe | at or above the 97th percentile |

The operational score is a hybrid of the BiLSTM class probabilities and a transparent GloFAS discharge prior (`HYBRID_MODEL_WEIGHT=0.4` by default):

- `0–39` Normal
- `40–64` Watch
- `65–84` Warning
- `85–100` Severe

Email alerts fire when the ensemble reaches **Warning** within the next **3 days**, with a **12-hour cooldown** per basin. All three are configurable.

### Model

- Two-layer bidirectional LSTM, hidden size 64, dropout 0.25.
- Learned 8-dimensional basin embedding added to every time step.
- Additive temporal attention pooling.
- 14-day input sequence of 19 engineered features.
- Chronological train / validation / test split — test data is strictly later in time.
- Class-weighted cross-entropy to counter the rarity of Warning/Severe days.

Current artifact on this machine:

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 83.6% | 88.4% |
| Macro-F1 | 64.8% | 63.3% |

Macro-F1 matters more than accuracy here because Normal days dominate the dataset. The test matrix is available in the dashboard's model card.

---

## Quick start (Windows)

```powershell
cd C:\Users\Nerv\floodguard-india
.\run.ps1
```

If PowerShell blocks local scripts, run `powershell -ExecutionPolicy Bypass -File .\run.ps1` once.

`run.ps1` creates `.venv` and installs `requirements.txt` on first run, then starts the server. Open <http://127.0.0.1:8000>. Use `.\run.ps1 -Port 8080 -Open` to change the port and launch a browser.

### Manual start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The first startup refreshes all five basins. If `models/flood_lstm.pt` is missing and `AUTO_TRAIN=true`, the server trains a model automatically (a few minutes). The dashboard shows the hydrological fallback while that happens.

### Useful CLI commands

```powershell
python -m app.bootstrap --check     # verify model artifacts and show metrics
python -m app.bootstrap --train     # retrain from cached/live history
python -m app.bootstrap --train --force-data --epochs 60
python -m app.bootstrap --refresh   # fetch live data and update the dashboard cache
```

### Tests

The regression suite is offline and safe to run at any time. It temporarily swaps in a test database and forces email simulation, so it never messages real subscribers.

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

It covers the five-basin definitions, causal feature engineering, threshold labels, the saved BiLSTM artifact, live-window inference, subscription lifecycle, normal-vs-Warning alert evaluation, duplicate suppression, unsubscribe tokens and the rendered email.

---

## Email alerts

Copy `.env.example` to `.env`. With no SMTP settings the app runs in **simulation mode**:

- every message is rendered as HTML and saved under `data/outbox/`,
- the dashboard marks delivery as `Simulated`,
- the full subscribe → alert → log path is testable without credentials.

To send real mail, set:

```dotenv
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USER=you@example.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=alerts@yourdomain.com
SMTP_FROM_NAME=FloodGuard India
SMTP_USE_TLS=true
SMTP_USE_SSL=false
APP_BASE_URL=http://127.0.0.1:8000
```

Restart the server and the dashboard chip changes to **SMTP delivery**. Subscribers can unsubscribe from any email via a signed token link.

Common provider settings (use an app password, never your normal account password):

| Provider | SMTP host | Port | TLS / SSL |
|---|---|---|---|
| Gmail | `smtp.gmail.com` | 587 | TLS on, SSL off |
| Microsoft 365 / Outlook | `smtp.office365.com` | 587 | TLS on, SSL off |
| Brevo | `smtp-relay.brevo.com` | 587 | TLS on, SSL off |
| Mailgun | `smtp.mailgun.org` | 587 | TLS on, SSL off |

Gmail requires 2-Step Verification and an App Password. The sending address (`SMTP_FROM`) must normally be verified by the provider. Keep `.env` private: it is already git-ignored.

Outside the monsoon season, use **Send demo warning** on the dashboard to preview the complete pipeline with a synthetic Warning. It is clearly labelled `DEMO` and does not imply a real flood.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard UI |
| GET | `/api/health` | Engine status, model mode, last refresh |
| GET | `/api/dashboard` | Full cached dashboard payload |
| POST | `/api/refresh` | Force a live fetch, inference and cache rebuild |
| GET | `/api/regions` | Basin definitions and active state |
| GET | `/api/regions/{id}` | Full payload for one basin |
| POST | `/api/subscribe` | `{ "email": "...", "regions": ["assam-brahmaputra"] }` |
| GET | `/api/unsubscribe?token=...` | Token-based unsubscribe page |
| GET | `/api/alerts` | Recent dispatch log |
| POST | `/api/alerts/test` | `{ "email": "...", "region_id": "..." }` |
| POST | `/api/alerts/demo` | Clearly-labelled synthetic Warning |

Interactive docs: <http://127.0.0.1:8000/docs>.

---

## Project layout

```
floodguard-india/
├─ app/
│  ├─ main.py          FastAPI routes, static assets, unsubscribe page
│  ├─ engine.py        refresh orchestration, dashboard cache, demo/test alerts
│  ├─ data.py          Open-Meteo fetch, cache, synthetic fallback
│  ├─ features.py      shared feature engineering, thresholds, scaling
│  ├─ model.py         FloodLSTM, training loop, artifacts
│  ├─ inference.py     hybrid prediction, series/forecast, drivers
│  ├─ alerts.py        subscriptions and alert evaluation
│  ├─ emails.py        HTML emails, SMTP and outbox simulation
│  ├─ db.py            SQLite subscribers, alerts, snapshots, predictions
│  ├─ config.py        environment, paths, risk constants
│  ├─ regions.py       the five basin definitions
│  └─ bootstrap.py     CLI: check / train / refresh
├─ tests/
│  └─ test_pipeline.py offline end-to-end regression tests
├─ web/
│  ├─ index.html       dashboard markup
│  ├─ styles.css       visual system and responsive rules
│  ├─ app.js           charts, polling, forms, basin detail, model card
│  ├─ favicon.svg
│  └─ vendor/chart.umd.min.js
├─ models/
│  ├─ flood_lstm.pt    trained weights
│  └─ metadata.json    metrics, thresholds, scalers, features
├─ data/               git-ignored runtime data (CSV cache, SQLite, outbox)
├─ requirements.txt
├─ .env.example
├─ run.ps1
└─ README.md
```

---

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `REFRESH_MINUTES` | `30` | Background refresh cadence |
| `AUTO_TRAIN` | `true` | Train automatically when the artifact is missing |
| `TRAIN_START_DATE` | `2019-01-01` | History start for training |
| `HISTORY_LAG_DAYS` | `7` | Safety lag before the archive end date |
| `ALERT_MIN_LEVEL` | `2` | 1=Watch, 2=Warning, 3=Severe |
| `ALERT_HORIZON_DAYS` | `3` | Forecast window inspected for alerts |
| `ALERT_COOLDOWN_HOURS` | `12` | Minimum gap between alerts per basin |
| `HYBRID_MODEL_WEIGHT` | `0.4` | BiLSTM share of the operational score |
| `APP_BASE_URL` | `http://localhost:8000` | Dashboard link used in emails |

---

## Notes and limitations

- The live weather endpoint exposes only a short rolling history; long-horizon learning uses the ERA5-Land archive.
- Warning-class recall is the hardest part of the task and the reason macro-F1 is the headline metric.
- GloFAS is a large-scale hydrological model, not a local gauge. It can lag or smooth small catchment spikes.
- The GloFAS prior is deliberately given 60% weight so that a confident LSTM cannot ignore an abnormal river signal.
- SQLite and the local CSV cache keep the project self-contained; there is no external queue or cloud dependency.
- For a real deployment, add authentication, a shared database, rate limiting and an official alerting agreement.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard shows `Cached data` | Open-Meteo was unreachable during the last sync. It retries automatically; use the refresh button. |
| `Demo fallback` status | No live data and no cache for that basin. Check the network and `data/live/`. |
| No email arrived | Without SMTP the app only simulates delivery — inspect `data/outbox/` and the dispatch log. |
| `model artifact missing` | Run `python -m app.bootstrap --train` or let `AUTO_TRAIN` finish on first start. |
| Port already in use | `.\run.ps1 -Port 8080` |

---

Built with FastAPI, PyTorch, Open-Meteo and Chart.js. Five basins, one shared horizon.
