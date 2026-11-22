# `intel-core`

Shared canonical lakehouse contracts, DuckDB/Parquet storage + atomic HITL overrides, unified Gemini Vision FinOps wrapper, media staging, pure-Python MMM simulation, and Golden Vision Eval benchmark for the **Creative & Marketing Intelligence Suite**.

## Modules

- `intel_core.schemas` — Canonical Pydantic v2 & PyArrow contracts (`dim_creative_asset`, `dim_creative_dissection`, `dim_human_overrides`, `silver_ad_delivery_raw`, `fct_ad_delivery_weekly`, `dim_creative_themes`, `fct_business_outcomes_weekly`, `gold_mmm_weekly_panel`, `gold_theme_spend_weekly`, `gold_brand_monthly`), brand/country ISO registry (`brands_registry.yaml`), and monthly FX conversion.
- `intel_core.storage` — Idempotent Hive-partitioned Parquet I/O, atomic UUID-drop-file `HumanOverrideStore`, and sub-50ms DuckDB `vw_live_creative_dissection` precedence resolution (`asset_id` > `media_hash_key`, latest `created_at` wins).
- `intel_core.gemini` — Thread-safe `GeminiMediaClient` (`thinking_budget=0`, XML `<untrusted_ad_context>` prompt fencing, Vertex AI + AI Studio parity) and `FinOpsGuard` (pre-flight + mid-batch `BudgetExceededError` circuit breakers + `costs_ledger.csv`).
- `intel_core.media` — `MediaStager` (URL canonicalization -> `asset_id`, 64-bit DCT pHash, optional `ffmpeg` 4-frame hero storyboard, deterministic `format_fit_score`).
- `intel_core.optimize` — Pure-Python `simulate_budget_reallocation()` over `response_curves.json` (zero `pymc` dependency for interactive FastAPI/ADK serving).
- `intel_core.evals` — Versioned YAML prompts + 50-asset offline EU Automotive Golden Vision Eval benchmark (`Brand F1`, `FPR`, `Hook Accuracy`, `$/1k assets`).

## Local Verification

```bash
uv sync --extra dev
uv run ruff check intel_core tests
uv run pytest
```

## Creative & Marketing Intelligence Suite (`v0.1.0`) — Module Map

`intel-core` is the shared contract & governance SDK for the 7-repository **`creative-intel-suite`** (flagship UI & copilot in [`brand-analytics-agent`](https://github.com/citycheah/brand-analytics-agent)):

| Component | Role in `creative-intel-suite` (`v0.1.0`) | Stack |
| :--- | :--- | :--- |
| **[`brand-analytics-agent`](https://github.com/citycheah/brand-analytics-agent)** | **Flagship Product Shell** — 5-page Next.js 15 Portal, FastAPI REST/SSE & Google ADK Copilot | Next.js 15, FastAPI, Google ADK, DuckDB |
| **[`intel-core`](https://github.com/citycheah/intel-core)** | **Core SDK** — Canonical contracts, DuckDB HITL views, FinOps, Hill optimizer, Golden Vision evals | Pydantic v2, PyArrow, DuckDB, Gemini |
| **[`intel-etl-spark`](https://github.com/citycheah/intel-etl-spark)** | **Medallion Batch ETL** — Bronze->Silver->Gold Spark jobs, EU Auto DGP & DuckDB local parity | PySpark 3.5, DuckDB, PyArrow |
| **[`intel-mmm`](https://github.com/citycheah/intel-mmm)** | **Causal Incrementality** — 3-tier Bayesian MMM ladder, separate theme attribution & ROAS guardrails | PyMC-Marketing, `nutpie`, Ridge |
| **[`creative-intelligence-toolkit`](https://github.com/citycheah/creative-intelligence-toolkit)** | **Ingestion Connectors** — Meta EU DSA Ad Library, Google Trends, YouTube/TikTok/IG/FB & alerts | Gemini, Playwright, `yt-dlp`, `intel-core` |
| **[`creative-analytics-grouping`](https://github.com/citycheah/creative-analytics-grouping)** | **Theme & Fatigue Engine** — Anti-join unseen dissection, 30-D UMAP+HDBSCAN & wearout tracking | Vertex/Gemini, HDBSCAN, UMAP |
| **[`competitive-velocity-ml-engine`](https://github.com/citycheah/competitive-velocity-ml-engine)** | **Velocity & Drift Engine** — YouTube $t_0$ breakout prediction, dbt PIT features & PSI drift | dbt-duckdb, LightGBM, SHAP |
