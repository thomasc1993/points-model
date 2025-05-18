# points-model

# 🏀 nba-pp-lgbm
Predict how many points an NBA / WNBA / G-League player will score **tonight** by mining
play-by-play data with **pbpstats** and training a gradient-boosted tree model
(**LightGBM**).

---

## 📑 Table of Contents
1. [Project philosophy](#project-philosophy)  
2. [Folder layout](#folder-layout)  
3. [Quick-start (two-minute demo)](#quick-start)  
4. [Step-by-step pipeline](#step-by-step-pipeline)  
5. [Feature cookbook](#feature-cookbook)  
6. [Training & evaluation](#training--evaluation)  
7. [Predicting upcoming games](#predicting-upcoming-games)  
8. [Troubleshooting](#troubleshooting)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## Project philosophy
* **Re-produce** published DFS / sports-betting signals without spending
  months wiring REST calls by hand – let *pbpstats* do the heavy lift.  
* **Keep data lineage explicit**: every numbered script writes a single
  parquet file; nothing else reads raw JSON.  
* **Stat-engineering first, ML second**: the model is intentionally simple
  so you can see which basketball stats actually move the needle.  

---

## Folder layout
```text
nba-pp-lgbm/
│
├── data/                     # generated artefacts only
│   ├── raw/                  # pbpstats caches JSON here
│   ├── interim/              # tidy parquet tables per object
│   └── features/             # one row per player-game
│
├── notebooks/                # optional ad-hoc EDA
├── src/                      # numbered ETL & ML scripts
│   ├── 00_fetch_season.py          ← download & cache API JSON
│   ├── 01_flatten_objects.py       ← JSON ➜ tidy parquet
│   ├── 02_player_game_table.py     ← boxscore + minutes
│   ├── 03_roll_windows.py          ← recent-form windows
│   ├── 04_on_off_lineups.py        ← synergy / WOWY splits
│   ├── 05_matchup_context.py       ← fatigue & opponent D
│   ├── 06_shot_profile.py          ← distance / zone mix
│   ├── 07_merge_features.py        ← union of engineered cols
│   ├── 08_train_lightgbm.py        ← k-fold + Optuna
│   ├── 09_eval_report.py           ← SHAP & gain charts
│   └── 10_predict_upcoming.py      ← tonight’s point calls
│
├── .env.example              # put your data dir & season flags here
└── requirements.txt
```

---

## Quick-start
```bash
# 0. clone repo & install deps
git clone https://github.com/yourhandle/nba-pp-lgbm.git
cd nba-pp-lgbm
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # installs pbpstats, lightgbm, etc.

# 1. choose a season in .env (defaults to 2023-24)
cp .env.example .env                     # then tweak if desired

# 2. run first two pipeline steps (≈5 min for one season)
python src/00_fetch_season.py
python src/01_flatten_objects.py

# 3. train a baseline model (≈1–2 min on CPU)
python src/08_train_lightgbm.py

# 4. project tonight’s slate
python src/10_predict_upcoming.py --date 2025-05-18
```

---

## Step-by-step pipeline
| Step | Script | Input | Output | Purpose |
|------|--------|-------|--------|---------|
| 0 |00_fetch_season.py|pbpstats API|`data/raw/*.json`|cache one season of Schedule, Boxscore, Possessions, Shots, EnhancedPbp|
| 1 |01_flatten_objects.py|raw JSON|`data/interim/{games,poss,shots}.parquet`|tidy, row-oriented tables|
| 2 |02_player_game_table.py|interim tables|`data/interim/player_games.parquet`|minutes + boxscore per player|
| 3 |03_roll_windows.py|player_games|rolling means / EWMAs|form & hot-hand|
| 4 |04_on_off_lineups.py|poss.parquet|WOWY parquet|synergy & pair effects|
| 5 |05_matchup_context.py|Schedule|context parquet|fatigue, road trip, DEF strength|
| 6 |06_shot_profile.py|shots.parquet|profile parquet|shot zones & style|
| 7 |07_merge_features.py|all of the above|`data/features/player_game.parquet`|master training matrix|
| 8 |08_train_lightgbm.py|features table|`models/` dir|model & Optuna params|
| 9 |09_eval_report.py|model + features|SHAP plots, csv|interpretation|
|10 |10_predict_upcoming.py|model + live schedule|csv / json|point projections|

Every script is **idempotent**; rerun after code tweaks without breaking
down-stream steps.

---

## Feature cookbook
A ready-made list (all engineered by the scripts above):

| Theme | Example columns |
|-------|-----------------|
| **Usage & involvement** | FGA, FTA, touches, sec-per-touch, USG %, assist-rate received |
| **Form** | rolling 3/5/10-game points, USG, eFG %, minutes; EWMA(α = 0.2) |
| **Synergy / on-off** | net-rating with top 3 teammates, pair assist % |
| **Shot profile** | mean distance, %rim, %mid, %corner-3, entropy, xPTS/shot |
| **Opponent defence** | DEF Rtg to date, rim/3PA conceded |
| **Schedule / fatigue** | rest days, is_b2b, road_trip_len, time_zone_jump |
| **Leverage split** | points in high- vs low-leverage possessions |
| **Context dummies** | home_flag, national_tv_flag, pre/post-ASB |
| **Interactions** | USG × pace, minutes × rest, opponent_DEF × shot_dist |

LightGBM will auto-discover many higher-order interactions; keep only those
that pop in SHAP or gain charts.

---

## Training & evaluation
```bash
python src/08_train_lightgbm.py          # 5-fold CV + Optuna
python src/09_eval_report.py             # gain.csv & shap/*.png
```

Outputs  
* `models/lgbm_points.pkl` – binary booster  
* `models/params.json` – tuned hyper-params  
* `reports/shap_summary.png`, `reports/gain.csv`

---

## Predicting upcoming games
```bash
python src/10_predict_upcoming.py --date 2025-05-18 \
       --season 2024-25 --phase "Playoffs" \
       --outfile projections_2025-05-18.csv
```
The script
1. pulls today’s schedule via `pbpstats.Schedule`,  
2. builds **only** features available before tip-off,  
3. outputs a CSV sorted by predicted points.

---

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `AttributeError: '.to_dict'` on event objects | Use `event.data` – every enhanced-PBP item exposes it. |
| Half-size event list (~450 rows) | Use `data_provider="stats_nba"` instead of `data_nba` for seasons ≤ 2017-18. |
| 429 / throttling from stats.nba.com | After step 0 set `"source":"file"` so later scripts stay offline. |
| Wrong timezone in rest-days calc | Schedule stores local arena date – convert with `pytz` before diffing. |

---

## Contributing
PRs welcome!  Please:

1. Open an issue first for any large feature.  
2. Format code with **black** & **isort**.  
3. Run `pytest` – tests must pass.  

---

## License
MIT – see [LICENSE](LICENSE) for details.

Enjoy exploring the play-by-play universe & may your MAE keep shrinking!
