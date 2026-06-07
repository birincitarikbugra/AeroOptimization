# AerOptimization

**Air Vehicle Design and Development — Project Scheduling with Uncertain Activity Durations**

EMU 406 System Analysis and Design II · Hacettepe University · 2025–2026  
Industry Partner: TUSAŞ (Turkish Aerospace)

Team: Tuğçe Başaran · Tarık Buğra Birinci · Zeynep Kızkapan · Koray Sarı  
Advisor: Assoc. Prof. Dr. Diclehan Tezcaner Öztürk

---

## What this project does

Air vehicle development programs involve hundreds of interdependent engineering activities. Traditional deterministic scheduling methods like CPM assume fixed activity durations and ignore rework loops, supply chain delays, and resource constraints. This leads to systematic underestimation of project completion time.

This project builds a layered scheduling model on top of a real 216-activity TUSAŞ program schedule:

1. **CPM baseline** — deterministic critical path, 3,099 days, 46 critical activities
2. **Monte Carlo simulation** — 10,000 iterations with three risk-class distributions (Triangular / Beta-PERT / Lognormal), phase-based correlation (Iman-Conover), rework loops, and supply chain delays → P80: 3,507 days
3. **RCPSP** — resource-constrained scheduling (Serial SGS + LFT heuristic) → P80 with lean staffing: 4,935 days
4. **Risk metrics** — Criticality Index (CI), Cruciality Index (CRI), Schedule Sensitivity Index (SSI)
5. **Mitigation scenarios** — four strategies compared under three risk worlds (Min-Max Regret) → S2 (resource boost) dominant, P80: 3,412 days

## Repository structure

```
AerOptimization/
├── index.html          # Interactive results dashboard (GitHub Pages)
├── model/
│   └── v2_main.py      # Full simulation model (Python)
├── requirements.txt    # Python dependencies
└── README.md
```

## Dashboard

The `index.html` file is a standalone static page with all simulation results embedded. No server needed — open it in any browser or view it via GitHub Pages.

## Running the model

```bash
pip install -r requirements.txt

# Place labeled_tasks.pkl in outputs/ first (TUSAŞ data, not included)
python model/v2_main.py
```

> The raw project data is not included in this repository for confidentiality reasons. The dashboard (index.html) contains all computed results.

## Key results

| Method | P80 (days) |
|---|---|
| CPM (deterministic) | 3,099 |
| Monte Carlo — full model | 3,507 |
| RCPSP — lean staffing | 4,935 |
| RCPSP — baseline staffing | 4,214 |
| RCPSP — S2 optimized | 3,412 |

The gap between the CPM estimate and the resource-constrained result is driven mainly by staffing constraints, not technical uncertainty.
