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
