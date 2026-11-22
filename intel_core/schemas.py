"""Canonical Pydantic v2 & PyArrow data contracts, brand/country registry, and FX helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pyarrow as pa
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_BRANDS_REGISTRY_PATH = CONFIGS_DIR / "brands_registry.yaml"

_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "ttclid",
    "msclkid",
    "igshid",
    "igsh",
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
    "ref",
    "ref_src",
}

SourceConnector = Literal[
    "meta_ad_lib_eu",
    "tiktok_cc",
    "synthetic_eu_auto",
    "youtube_organic",
    "apify",
]
MediaType = Literal["video", "image", "text"]
HookType = Literal[
    "offer_first",
    "ugc_testimonial",
    "problem_agitation",
    "product_demo",
    "brand_story",
    "unknown",
]
OverrideTargetType = Literal["asset_id", "media_hash_key", "theme_label"]
OverrideFieldName = Literal["belongs_to_brand", "matched_brand_id", "theme_label", "category"]
AuditStatus = Literal[
    "domain_verified",
    "vision_verified",
    "human_override",
    "unverified_tail",
    "rejected",
]
ThemeScope = Literal["shared_battleground", "brand_exclusive_whitespace"]
OutcomeMetricName = Literal["conversions", "revenue", "google_trends_index"]
OutcomeSource = Literal["synthetic_dgp", "google_trends_proxy", "public_benchmark"]


@lru_cache(maxsize=4)
def load_brands_registry(registry_path: str | None = None) -> dict[str, Any]:
    """Loads the canonical brands, countries, and monthly FX registry YAML."""
    resolved = Path(
        registry_path
        or os.environ.get("INTEL_BRANDS_REGISTRY_PATH")
        or DEFAULT_BRANDS_REGISTRY_PATH
    )
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _country_lookup_tables(
    registry_path: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    reg = load_brands_registry(registry_path)
    countries = reg.get("countries", {})
    alias_to_iso: dict[str, str] = {}
    iso_to_display: dict[str, str] = {}
    for iso, meta in countries.items():
        iso_upper = str(iso).upper().strip()
        display = str(meta.get("display_name", iso_upper)).strip()
        iso_to_display[iso_upper] = display
        alias_to_iso[iso_upper.lower()] = iso_upper
        alias_to_iso[display.lower()] = iso_upper
        for alias in meta.get("aliases", []):
            alias_to_iso[str(alias).strip().lower()] = iso_upper
    return alias_to_iso, iso_to_display


def normalize_country_iso(country_or_iso: str, registry_path: str | None = None) -> str:
    """Normalizes country display names or aliases (e.g. 'Germany', 'UK') to ISO-2 ('DE', 'GB')."""
    raw = str(country_or_iso or "").strip()
    if not raw:
        raise ValueError("Country value cannot be empty.")
    alias_to_iso, _ = _country_lookup_tables(registry_path)
    iso = alias_to_iso.get(raw.lower())
    if not iso:
        raise ValueError(f"Unsupported or unknown country: {country_or_iso!r}")
    return iso


def country_display_name(country_iso_or_name: str, registry_path: str | None = None) -> str:
    """Resolves ISO-2 or alias to canonical display name (e.g. 'DE' -> 'Germany')."""
    iso = normalize_country_iso(country_iso_or_name, registry_path=registry_path)
    _, iso_to_display = _country_lookup_tables(registry_path)
    return iso_to_display[iso]


def supported_country_isos(registry_path: str | None = None) -> set[str]:
    _, iso_to_display = _country_lookup_tables(registry_path)
    return set(iso_to_display.keys())


def supported_country_names(registry_path: str | None = None) -> set[str]:
    _, iso_to_display = _country_lookup_tables(registry_path)
    return set(iso_to_display.values())


def canonicalize_brand_id(raw_name: str, registry_path: str | None = None) -> str | None:
    """Resolves a raw brand string or alias to its canonical registry brand_id."""
    cleaned = " ".join(str(raw_name or "").strip().lower().split())
    if not cleaned:
        return None
    reg = load_brands_registry(registry_path)
    for brand_id, profile in (reg.get("brands") or {}).items():
        if cleaned == str(brand_id).lower():
            return str(profile.get("brand_id", brand_id))
        aliases = [str(a).strip().lower() for a in profile.get("aliases", [])]
        if cleaned in aliases or cleaned == str(profile.get("display_name", "")).lower():
            return str(profile.get("brand_id", brand_id))
    return None


def get_brand_profile(brand_id: str, registry_path: str | None = None) -> dict[str, Any]:
    reg = load_brands_registry(registry_path)
    brands = reg.get("brands") or {}
    if brand_id in brands:
        return dict(brands[brand_id])
    canon = canonicalize_brand_id(brand_id, registry_path=registry_path)
    if canon and canon in brands:
        return dict(brands[canon])
    raise KeyError(f"Unknown canonical brand_id: {brand_id!r}")


def list_vertical_brands(
    vertical: str = "EU_AUTOMOTIVE", registry_path: str | None = None
) -> list[str]:
    reg = load_brands_registry(registry_path)
    return [
        str(profile.get("brand_id", key))
        for key, profile in (reg.get("brands") or {}).items()
        if str(profile.get("vertical", "")).upper() == vertical.upper()
    ]


def convert_to_usd(
    amount_local: float,
    currency: str,
    year_month: str | dt.date | dt.datetime | None = None,
    registry_path: str | None = None,
) -> float:
    """Converts local currency spend to USD using monthly registry FX rates."""
    cur = str(currency or "USD").strip().upper()
    reg = load_brands_registry(registry_path)
    rates_map = (reg.get("fx_rates_to_usd") or {}).get(cur)
    if not rates_map:
        raise ValueError(f"Unsupported currency in FX table: {cur!r}")

    ym_key: str | None = None
    if isinstance(year_month, dt.datetime | dt.date):
        ym_key = year_month.strftime("%Y-%m")
    elif isinstance(year_month, str) and len(year_month.strip()) >= 7:
        ym_key = year_month.strip()[:7]

    if ym_key and ym_key in rates_map:
        rate = float(rates_map[ym_key])
    else:
        dated_keys = sorted(k for k in rates_map if k != "default" and (not ym_key or k <= ym_key))
        if dated_keys:
            rate = float(rates_map[dated_keys[-1]])
        else:
            rate = float(rates_map["default"])
    return round(float(amount_local) * rate, 6)


def canonicalize_media_url(uri: str) -> str:
    """Strips volatile tracking query params and normalizes scheme/host for stable asset hashing."""
    raw = str(uri or "").strip()
    if not raw:
        raise ValueError("media_uri cannot be empty.")
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https", "gs"):
        kept_params = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not (k.lower().startswith(_TRACKING_QUERY_PREFIXES) or k.lower() in _TRACKING_QUERY_KEYS)
        ]
        kept_params.sort()
        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            query=urlencode(kept_params, doseq=True),
            fragment="",
        )
        return urlunparse(normalized)
    # Local filesystem path: normalize separators without losing relative clarity
    return Path(raw).as_posix()


def compute_asset_id(media_uri: str) -> str:
    """Deterministic SHA-256 asset_id derived from canonicalized media URI."""
    canon = canonicalize_media_url(media_uri)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def compute_media_hash_key(phash: str | None, asset_id: str) -> str:
    """SQL-equivalent of COALESCE(NULLIF(TRIM(phash), ''), asset_id)."""
    cleaned = (phash or "").strip()
    return cleaned if cleaned else asset_id


def iso_week_start(value: dt.date | dt.datetime | str) -> dt.date:
    """Returns the ISO Monday DATE for any date, datetime, or ISO string."""
    if isinstance(value, str):
        parsed_date = dt.date.fromisoformat(value[:10])
    elif isinstance(value, dt.datetime):
        parsed_date = value.date()
    else:
        parsed_date = value
    return parsed_date - dt.timedelta(days=parsed_date.weekday())


def _parse_iso_date(v: Any) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, str):
        return dt.date.fromisoformat(v[:10])
    raise ValueError(f"Invalid date value: {v!r}")


def _ensure_utc(v: Any) -> dt.datetime | None:
    if v is None:
        return None
    if isinstance(v, str):
        parsed = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    elif isinstance(v, dt.datetime):
        parsed = v
    else:
        raise ValueError(f"Invalid datetime value: {v!r}")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


class DimCreativeAsset(BaseModel):
    """Canonical asset registry table (`dim_creative_asset`, PK: `asset_id`)."""

    asset_id: str = Field(default="")
    phash: str | None = None
    media_hash_key: str = Field(default="")
    brand_id_raw: str
    source_connector: SourceConnector
    media_type: MediaType
    media_uri: str
    aspect_ratio: str | None = None
    duration_sec: float | None = Field(default=None, ge=0.0)
    first_seen_date: dt.date
    last_seen_date: dt.date
    is_active: bool = True

    @model_validator(mode="after")
    def _populate_and_validate_keys(self) -> DimCreativeAsset:
        if not self.asset_id:
            self.asset_id = compute_asset_id(self.media_uri)
        expected_key = compute_media_hash_key(self.phash, self.asset_id)
        if not self.media_hash_key:
            self.media_hash_key = expected_key
        elif self.media_hash_key != expected_key:
            raise ValueError(
                f"media_hash_key {self.media_hash_key!r} violates COALESCE(NULLIF(phash,''), asset_id) == {expected_key!r}"
            )
        if self.last_seen_date < self.first_seen_date:
            raise ValueError("last_seen_date cannot precede first_seen_date.")
        return self


class DissectionLLMOutput(BaseModel):
    """Structured JSON schema returned directly by Gemini Vision dissection."""

    belongs_to_brand: bool
    matched_brand_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    category: str
    core_concept_summary: str
    messaging_angle: str
    visual_style_tags: list[str] = Field(default_factory=list)
    hook_type: HookType = "unknown"
    hook_transcript: str = ""
    time_to_brand_sec: float | None = Field(default=None, ge=0.0)
    has_voiceover: bool = False
    has_captions: bool = False
    cta_text: str = ""
    cta_clarity_score: int = Field(default=1, ge=1, le=5)


class DimCreativeDissection(DissectionLLMOutput):
    """Lakehouse creative dissection table (`dim_creative_dissection`, PK: `(media_hash_key, evaluated_brand_id, prompt_version)`)."""

    media_hash_key: str
    evaluated_brand_id: str
    prompt_version: str
    is_active_version: bool = True
    model_name: str
    format_fit_score: float = Field(default=0.0, ge=0.0, le=100.0)


class DimHumanOverride(BaseModel):
    """Human-in-the-loop override table (`dim_human_overrides`, PK: `override_id`)."""

    override_id: str
    target_key_type: OverrideTargetType
    target_key_value: str
    field_name: OverrideFieldName
    ai_value: str | None = None
    human_value: str
    reviewer: str
    reason: str = ""
    created_at: dt.datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_created_at(cls, v: Any) -> dt.datetime:
        res = _ensure_utc(v)
        assert res is not None
        return res


class SilverAdDeliveryRaw(BaseModel):
    """Pre-attribution raw occurrence grain (`silver_ad_delivery_raw`, PK: `occurrence_id`)."""

    occurrence_id: str
    week_start: dt.date
    country_iso: str
    asset_id: str
    media_hash_key: str
    channel_group: str
    publisher_name: str
    campaign_id: str
    candidate_brand_ids: list[str] = Field(default_factory=list)
    raw_impressions: int = Field(ge=0)
    raw_reach_est: int = Field(ge=0)
    raw_eu_total_reach: int | None = Field(default=None, ge=0)
    snapshot_ts: dt.datetime | None = None
    is_weekly_interpolated: bool = False
    raw_spend_local: float = Field(ge=0.0)
    currency: str
    raw_spend_usd: float = Field(ge=0.0)
    is_spend_modeled: bool = False

    @field_validator("snapshot_ts", mode="before")
    @classmethod
    def _normalize_snapshot_ts(cls, v: Any) -> dt.datetime | None:
        return _ensure_utc(v)

    @field_validator("country_iso")
    @classmethod
    def _validate_country_iso(cls, v: str) -> str:
        return normalize_country_iso(v)

    @field_validator("week_start")
    @classmethod
    def _validate_monday(cls, v: dt.date) -> dt.date:
        d = _parse_iso_date(v)
        if d.weekday() != 0:
            raise ValueError(f"week_start must be an ISO Monday, got {d} (weekday={d.weekday()})")
        return d


class FctAdDeliveryWeekly(BaseModel):
    """Post-verification, post-override 1/K attributed Gold fact (`fct_ad_delivery_weekly`, PK: `(occurrence_id, brand_id)`)."""

    week_start: dt.date
    country_iso: str
    brand_id: str
    occurrence_id: str
    asset_id: str
    media_hash_key: str
    channel_group: str
    publisher_name: str
    campaign_id: str
    attribution_k: int = Field(ge=1)
    spend_usd: float = Field(ge=0.0)
    impressions: float = Field(ge=0.0)
    reach_est: float = Field(ge=0.0)
    is_spend_modeled: bool = False
    is_weekly_interpolated: bool = False
    audit_status: AuditStatus

    @field_validator("country_iso")
    @classmethod
    def _validate_country_iso(cls, v: str) -> str:
        return normalize_country_iso(v)

    @field_validator("week_start")
    @classmethod
    def _validate_monday(cls, v: dt.date) -> dt.date:
        d = _parse_iso_date(v)
        if d.weekday() != 0:
            raise ValueError(f"week_start must be an ISO Monday, got {d}")
        return d


class DimCreativeTheme(BaseModel):
    """Dynamic joint clustering output (`dim_creative_themes`, PK: `(media_hash_key, run_id)`)."""

    media_hash_key: str
    run_id: str
    is_active_run: bool = True
    theme_id: str
    theme_label: str
    theme_scope: ThemeScope
    umap_x: float
    umap_y: float
    centroid_similarity: float = Field(ge=-1.0, le=1.0)


class FctBusinessOutcomesWeekly(BaseModel):
    """Weekly KPI / proxy outcome series (`fct_business_outcomes_weekly`, PK: `(week_start, country_iso, brand_id, outcome_metric_name)`)."""

    week_start: dt.date
    country_iso: str
    brand_id: str
    outcome_metric_name: OutcomeMetricName
    outcome_value: float
    outcome_source: OutcomeSource
    promo_flag: int = Field(default=0, ge=0, le=1)
    holiday_flag: int = Field(default=0, ge=0, le=1)
    price_index: float = Field(default=1.0, gt=0.0)
    macro_seasonality: float = 0.0

    @field_validator("country_iso")
    @classmethod
    def _validate_country_iso(cls, v: str) -> str:
        return normalize_country_iso(v)

    @field_validator("week_start")
    @classmethod
    def _validate_monday(cls, v: dt.date) -> dt.date:
        d = _parse_iso_date(v)
        if d.weekday() != 0:
            raise ValueError(f"week_start must be an ISO Monday, got {d}")
        return d


class GoldMmmWeeklyPanel(BaseModel):
    """Channel MMM regression panel (`gold_mmm_weekly_panel`, PK: `(week_start, country_iso, brand_id)`).

    Excludes `competitor_sov_pct` (collider) and theme spend columns (compositional collinearity).
    Drops `own_impressions_total` when `is_spend_modeled=True`.
    """

    week_start: dt.date
    country_iso: str
    brand_id: str
    own_spend_meta_usd: float = Field(default=0.0, ge=0.0)
    own_spend_youtube_usd: float = Field(default=0.0, ge=0.0)
    own_spend_tiktok_usd: float = Field(default=0.0, ge=0.0)
    own_spend_display_usd: float = Field(default=0.0, ge=0.0)
    own_spend_total_usd: float = Field(default=0.0, ge=0.0)
    own_impressions_total: float | None = Field(default=None, ge=0.0)
    competitor_spend_usd: float = Field(default=0.0, ge=0.0)
    competitor_impressions: float = Field(default=0.0, ge=0.0)
    category_hhi: float = Field(default=0.0, ge=0.0, le=10000.0)
    avg_creative_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_creative_age_weeks: float = Field(default=0.0, ge=0.0)
    audit_coverage_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    is_spend_modeled: bool = False
    is_weekly_interpolated: bool = False
    outcome_metric_name: OutcomeMetricName = "conversions"
    outcome_value: float = 0.0
    outcome_source: OutcomeSource = "synthetic_dgp"
    promo_flag: int = Field(default=0, ge=0, le=1)
    holiday_flag: int = Field(default=0, ge=0, le=1)
    price_index: float = Field(default=1.0, gt=0.0)
    macro_seasonality: float = 0.0

    @field_validator("country_iso")
    @classmethod
    def _validate_country_iso(cls, v: str) -> str:
        return normalize_country_iso(v)

    @field_validator("week_start")
    @classmethod
    def _validate_monday(cls, v: dt.date) -> dt.date:
        d = _parse_iso_date(v)
        if d.weekday() != 0:
            raise ValueError(f"week_start must be an ISO Monday, got {d}")
        return d

    @model_validator(mode="after")
    def _enforce_modeled_spend_identifiability(self) -> GoldMmmWeeklyPanel:
        if self.is_spend_modeled and self.own_impressions_total is not None:
            raise ValueError(
                "own_impressions_total must be None when is_spend_modeled=True to prevent exact scalar collinearity."
            )
        return self


class GoldThemeSpendWeekly(BaseModel):
    """Structurally separate theme attribution fact (`gold_theme_spend_weekly`, PK: `(week_start, country_iso, brand_id, theme_label)`)."""

    week_start: dt.date
    country_iso: str
    brand_id: str
    theme_id: str
    theme_label: str
    theme_spend_usd: float = Field(ge=0.0)
    theme_impressions: float = Field(ge=0.0)

    @field_validator("country_iso")
    @classmethod
    def _validate_country_iso(cls, v: str) -> str:
        return normalize_country_iso(v)

    @field_validator("week_start")
    @classmethod
    def _validate_monday(cls, v: dt.date) -> dt.date:
        d = _parse_iso_date(v)
        if d.weekday() != 0:
            raise ValueError(f"week_start must be an ISO Monday, got {d}")
        return d


class GoldBrandMonthly(BaseModel):
    """Monthly SOS/SOV summary (`gold_brand_monthly`, PK: `(month_start, country_iso, brand_id)`)."""

    month_start: dt.date
    country_iso: str
    brand_id: str
    spend_usd: float = Field(ge=0.0)
    impressions: float = Field(ge=0.0)
    sos_pct: float = Field(ge=0.0, le=100.0)
    sov_pct: float = Field(ge=0.0, le=100.0)
    audit_coverage_pct: float = Field(ge=0.0, le=1.0)
    creative_count: int = Field(ge=0)

    @field_validator("country_iso")
    @classmethod
    def _validate_country_iso(cls, v: str) -> str:
        return normalize_country_iso(v)


ARROW_SCHEMAS: dict[str, pa.Schema] = {
    "dim_creative_asset": pa.schema(
        [
            pa.field("asset_id", pa.string(), nullable=False),
            pa.field("phash", pa.string(), nullable=True),
            pa.field("media_hash_key", pa.string(), nullable=False),
            pa.field("brand_id_raw", pa.string(), nullable=False),
            pa.field("source_connector", pa.string(), nullable=False),
            pa.field("media_type", pa.string(), nullable=False),
            pa.field("media_uri", pa.string(), nullable=False),
            pa.field("aspect_ratio", pa.string(), nullable=True),
            pa.field("duration_sec", pa.float64(), nullable=True),
            pa.field("first_seen_date", pa.date32(), nullable=False),
            pa.field("last_seen_date", pa.date32(), nullable=False),
            pa.field("is_active", pa.bool_(), nullable=False),
        ]
    ),
    "dim_creative_dissection": pa.schema(
        [
            pa.field("media_hash_key", pa.string(), nullable=False),
            pa.field("evaluated_brand_id", pa.string(), nullable=False),
            pa.field("prompt_version", pa.string(), nullable=False),
            pa.field("is_active_version", pa.bool_(), nullable=False),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("belongs_to_brand", pa.bool_(), nullable=False),
            pa.field("matched_brand_id", pa.string(), nullable=True),
            pa.field("confidence", pa.float64(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("core_concept_summary", pa.string(), nullable=False),
            pa.field("messaging_angle", pa.string(), nullable=False),
            pa.field("visual_style_tags", pa.list_(pa.string()), nullable=False),
            pa.field("hook_type", pa.string(), nullable=False),
            pa.field("hook_transcript", pa.string(), nullable=False),
            pa.field("time_to_brand_sec", pa.float64(), nullable=True),
            pa.field("has_voiceover", pa.bool_(), nullable=False),
            pa.field("has_captions", pa.bool_(), nullable=False),
            pa.field("cta_text", pa.string(), nullable=False),
            pa.field("cta_clarity_score", pa.int32(), nullable=False),
            pa.field("format_fit_score", pa.float64(), nullable=False),
        ]
    ),
    "dim_human_overrides": pa.schema(
        [
            pa.field("override_id", pa.string(), nullable=False),
            pa.field("target_key_type", pa.string(), nullable=False),
            pa.field("target_key_value", pa.string(), nullable=False),
            pa.field("field_name", pa.string(), nullable=False),
            pa.field("ai_value", pa.string(), nullable=True),
            pa.field("human_value", pa.string(), nullable=False),
            pa.field("reviewer", pa.string(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        ]
    ),
    "silver_ad_delivery_raw": pa.schema(
        [
            pa.field("occurrence_id", pa.string(), nullable=False),
            pa.field("week_start", pa.date32(), nullable=False),
            pa.field("country_iso", pa.string(), nullable=False),
            pa.field("asset_id", pa.string(), nullable=False),
            pa.field("media_hash_key", pa.string(), nullable=False),
            pa.field("channel_group", pa.string(), nullable=False),
            pa.field("publisher_name", pa.string(), nullable=False),
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("candidate_brand_ids", pa.list_(pa.string()), nullable=False),
            pa.field("raw_impressions", pa.int64(), nullable=False),
            pa.field("raw_reach_est", pa.int64(), nullable=False),
            pa.field("raw_eu_total_reach", pa.int64(), nullable=True),
            pa.field("snapshot_ts", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("is_weekly_interpolated", pa.bool_(), nullable=False),
            pa.field("raw_spend_local", pa.float64(), nullable=False),
            pa.field("currency", pa.string(), nullable=False),
            pa.field("raw_spend_usd", pa.float64(), nullable=False),
            pa.field("is_spend_modeled", pa.bool_(), nullable=False),
        ]
    ),
    "fct_ad_delivery_weekly": pa.schema(
        [
            pa.field("week_start", pa.date32(), nullable=False),
            pa.field("country_iso", pa.string(), nullable=False),
            pa.field("brand_id", pa.string(), nullable=False),
            pa.field("occurrence_id", pa.string(), nullable=False),
            pa.field("asset_id", pa.string(), nullable=False),
            pa.field("media_hash_key", pa.string(), nullable=False),
            pa.field("channel_group", pa.string(), nullable=False),
            pa.field("publisher_name", pa.string(), nullable=False),
            pa.field("campaign_id", pa.string(), nullable=False),
            pa.field("attribution_k", pa.int32(), nullable=False),
            pa.field("spend_usd", pa.float64(), nullable=False),
            pa.field("impressions", pa.float64(), nullable=False),
            pa.field("reach_est", pa.float64(), nullable=False),
            pa.field("is_spend_modeled", pa.bool_(), nullable=False),
            pa.field("is_weekly_interpolated", pa.bool_(), nullable=False),
            pa.field("audit_status", pa.string(), nullable=False),
        ]
    ),
    "dim_creative_themes": pa.schema(
        [
            pa.field("media_hash_key", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("is_active_run", pa.bool_(), nullable=False),
            pa.field("theme_id", pa.string(), nullable=False),
            pa.field("theme_label", pa.string(), nullable=False),
            pa.field("theme_scope", pa.string(), nullable=False),
            pa.field("umap_x", pa.float64(), nullable=False),
            pa.field("umap_y", pa.float64(), nullable=False),
            pa.field("centroid_similarity", pa.float64(), nullable=False),
        ]
    ),
    "fct_business_outcomes_weekly": pa.schema(
        [
            pa.field("week_start", pa.date32(), nullable=False),
            pa.field("country_iso", pa.string(), nullable=False),
            pa.field("brand_id", pa.string(), nullable=False),
            pa.field("outcome_metric_name", pa.string(), nullable=False),
            pa.field("outcome_value", pa.float64(), nullable=False),
            pa.field("outcome_source", pa.string(), nullable=False),
            pa.field("promo_flag", pa.int32(), nullable=False),
            pa.field("holiday_flag", pa.int32(), nullable=False),
            pa.field("price_index", pa.float64(), nullable=False),
            pa.field("macro_seasonality", pa.float64(), nullable=False),
        ]
    ),
    "gold_mmm_weekly_panel": pa.schema(
        [
            pa.field("week_start", pa.date32(), nullable=False),
            pa.field("country_iso", pa.string(), nullable=False),
            pa.field("brand_id", pa.string(), nullable=False),
            pa.field("own_spend_meta_usd", pa.float64(), nullable=False),
            pa.field("own_spend_youtube_usd", pa.float64(), nullable=False),
            pa.field("own_spend_tiktok_usd", pa.float64(), nullable=False),
            pa.field("own_spend_display_usd", pa.float64(), nullable=False),
            pa.field("own_spend_total_usd", pa.float64(), nullable=False),
            pa.field("own_impressions_total", pa.float64(), nullable=True),
            pa.field("competitor_spend_usd", pa.float64(), nullable=False),
            pa.field("competitor_impressions", pa.float64(), nullable=False),
            pa.field("category_hhi", pa.float64(), nullable=False),
            pa.field("avg_creative_quality_score", pa.float64(), nullable=False),
            pa.field("avg_creative_age_weeks", pa.float64(), nullable=False),
            pa.field("audit_coverage_pct", pa.float64(), nullable=False),
            pa.field("is_spend_modeled", pa.bool_(), nullable=False),
            pa.field("is_weekly_interpolated", pa.bool_(), nullable=False),
            pa.field("outcome_metric_name", pa.string(), nullable=False),
            pa.field("outcome_value", pa.float64(), nullable=False),
            pa.field("outcome_source", pa.string(), nullable=False),
            pa.field("promo_flag", pa.int32(), nullable=False),
            pa.field("holiday_flag", pa.int32(), nullable=False),
            pa.field("price_index", pa.float64(), nullable=False),
            pa.field("macro_seasonality", pa.float64(), nullable=False),
        ]
    ),
    "gold_theme_spend_weekly": pa.schema(
        [
            pa.field("week_start", pa.date32(), nullable=False),
            pa.field("country_iso", pa.string(), nullable=False),
            pa.field("brand_id", pa.string(), nullable=False),
            pa.field("theme_id", pa.string(), nullable=False),
            pa.field("theme_label", pa.string(), nullable=False),
            pa.field("theme_spend_usd", pa.float64(), nullable=False),
            pa.field("theme_impressions", pa.float64(), nullable=False),
        ]
    ),
    "gold_brand_monthly": pa.schema(
        [
            pa.field("month_start", pa.date32(), nullable=False),
            pa.field("country_iso", pa.string(), nullable=False),
            pa.field("brand_id", pa.string(), nullable=False),
            pa.field("spend_usd", pa.float64(), nullable=False),
            pa.field("impressions", pa.float64(), nullable=False),
            pa.field("sos_pct", pa.float64(), nullable=False),
            pa.field("sov_pct", pa.float64(), nullable=False),
            pa.field("audit_coverage_pct", pa.float64(), nullable=False),
            pa.field("creative_count", pa.int32(), nullable=False),
        ]
    ),
}

TABLE_MODELS: dict[str, type[BaseModel]] = {
    "dim_creative_asset": DimCreativeAsset,
    "dim_creative_dissection": DimCreativeDissection,
    "dim_human_overrides": DimHumanOverride,
    "silver_ad_delivery_raw": SilverAdDeliveryRaw,
    "fct_ad_delivery_weekly": FctAdDeliveryWeekly,
    "dim_creative_themes": DimCreativeTheme,
    "fct_business_outcomes_weekly": FctBusinessOutcomesWeekly,
    "gold_mmm_weekly_panel": GoldMmmWeeklyPanel,
    "gold_theme_spend_weekly": GoldThemeSpendWeekly,
    "gold_brand_monthly": GoldBrandMonthly,
}


def validate_records(table_name: str, rows: list[dict[str, Any] | BaseModel]) -> list[BaseModel]:
    """Validates a sequence of dicts or models against the canonical contract for `table_name`."""
    if table_name not in TABLE_MODELS:
        raise KeyError(f"Unknown canonical table: {table_name!r}")
    model_cls = TABLE_MODELS[table_name]
    return [r if isinstance(r, model_cls) else model_cls.model_validate(r) for r in rows]


def records_to_arrow(table_name: str, rows: list[dict[str, Any] | BaseModel]) -> pa.Table:
    """Validates records and materializes a strictly-typed PyArrow Table."""
    schema = ARROW_SCHEMAS[table_name]
    validated = validate_records(table_name, rows)
    dicts = [m.model_dump(mode="python") for m in validated]
    if not dicts:
        return pa.Table.from_pylist([], schema=schema)
    return pa.Table.from_pylist(dicts, schema=schema)
