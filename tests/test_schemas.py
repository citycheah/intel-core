import datetime as dt

import pytest
from pydantic import ValidationError

from intel_core.schemas import (
    ARROW_SCHEMAS,
    DimCreativeAsset,
    GoldMmmWeeklyPanel,
    canonicalize_brand_id,
    canonicalize_media_url,
    compute_asset_id,
    compute_media_hash_key,
    convert_to_usd,
    country_display_name,
    iso_week_start,
    list_vertical_brands,
    normalize_country_iso,
    records_to_arrow,
)


def test_country_iso_and_brand_canonicalization():
    assert normalize_country_iso("Germany") == "DE"
    assert normalize_country_iso("Deutschland") == "DE"
    assert normalize_country_iso("uk") == "GB"
    assert country_display_name("DE") == "Germany"
    assert country_display_name("gb") == "United Kingdom"

    assert canonicalize_brand_id("BMW Deutschland") == "BMW"
    assert canonicalize_brand_id("mercedes") == "Mercedes-Benz"
    assert canonicalize_brand_id("Audi AG") == "Audi"
    assert canonicalize_brand_id("porsche") == "Porsche"
    assert set(list_vertical_brands("EU_AUTOMOTIVE")) == {"BMW", "Mercedes-Benz", "Audi", "Porsche"}


def test_fx_conversion_and_week_alignment():
    assert convert_to_usd(100.0, "EUR", "2026-01-15") == pytest.approx(108.0)
    assert convert_to_usd(100.0, "GBP", dt.date(2026, 6, 10)) == pytest.approx(129.0)
    assert convert_to_usd(100.0, "USD", None) == pytest.approx(100.0)
    assert iso_week_start("2026-01-07") == dt.date(2026, 1, 5)


def test_asset_id_and_media_hash_key_prevent_download_failed_collapse():
    u1 = "https://cdn.example.com/ad1.mp4?fbclid=abc123&utm_source=meta&id=42"
    u2 = "https://CDN.EXAMPLE.COM/ad1.mp4?id=42&utm_medium=paid"
    assert canonicalize_media_url(u1) == canonicalize_media_url(u2)
    assert compute_asset_id(u1) == compute_asset_id(u2)

    a1 = DimCreativeAsset(
        brand_id_raw="BMW",
        source_connector="meta_ad_lib_eu",
        media_type="video",
        media_uri="https://cdn.example.com/ad1.mp4",
        phash=None,
        first_seen_date="2026-01-05",
        last_seen_date="2026-01-12",
    )
    a2 = DimCreativeAsset(
        brand_id_raw="BMW",
        source_connector="meta_ad_lib_eu",
        media_type="video",
        media_uri="https://cdn.example.com/ad2.mp4",
        phash="",
        first_seen_date="2026-01-05",
        last_seen_date="2026-01-12",
    )
    assert a1.media_hash_key == a1.asset_id
    assert a2.media_hash_key == a2.asset_id
    assert a1.media_hash_key != a2.media_hash_key
    assert compute_media_hash_key("ff00aa11", "asset_x") == "ff00aa11"


def test_mmm_panel_rejects_non_monday_and_modeled_spend_impressions():
    with pytest.raises(ValidationError, match="ISO Monday"):
        GoldMmmWeeklyPanel(
            week_start="2026-01-06",  # Tuesday
            country_iso="DE",
            brand_id="BMW",
        )

    with pytest.raises(ValidationError, match="own_impressions_total must be None"):
        GoldMmmWeeklyPanel(
            week_start="2026-01-05",
            country_iso="DE",
            brand_id="BMW",
            is_spend_modeled=True,
            own_impressions_total=500000.0,
        )


def test_arrow_empty_schema_materialization():
    for table_name, expected in ARROW_SCHEMAS.items():
        tbl = records_to_arrow(table_name, [])
        assert tbl.schema.equals(expected)
        assert tbl.num_rows == 0
