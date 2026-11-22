import datetime as dt
import time

import duckdb
import pandas as pd

from intel_core.schemas import DimCreativeAsset, DimCreativeDissection
from intel_core.storage import (
    HumanOverrideStore,
    LakehouseStore,
    read_parquet_table,
    register_live_dissection_view,
    write_parquet_table,
)


def test_hive_partitioned_parquet_roundtrip_and_overwrite(tmp_path):
    part_dir = tmp_path / "fct_ad_delivery_weekly"
    rows = [
        {
            "week_start": dt.date(2026, 1, 5),
            "country_iso": "DE",
            "brand_id": "BMW",
            "occurrence_id": "occ_1",
            "asset_id": "a1",
            "media_hash_key": "ph1",
            "channel_group": "meta_paid",
            "publisher_name": "Instagram",
            "campaign_id": "cmp_1",
            "attribution_k": 1,
            "spend_usd": 1200.0,
            "impressions": 100000.0,
            "reach_est": 75000.0,
            "is_spend_modeled": False,
            "is_weekly_interpolated": False,
            "audit_status": "vision_verified",
        },
        {
            "week_start": dt.date(2026, 1, 5),
            "country_iso": "FR",
            "brand_id": "Audi",
            "occurrence_id": "occ_2",
            "asset_id": "a2",
            "media_hash_key": "ph2",
            "channel_group": "meta_paid",
            "publisher_name": "Facebook",
            "campaign_id": "cmp_2",
            "attribution_k": 1,
            "spend_usd": 800.0,
            "impressions": 60000.0,
            "reach_est": 45000.0,
            "is_spend_modeled": False,
            "is_weekly_interpolated": False,
            "audit_status": "domain_verified",
        },
    ]
    write_parquet_table(
        rows,
        part_dir,
        table_name="fct_ad_delivery_weekly",
        partition_cols=["country_iso"],
    )
    # Re-write DE only -> should replace DE cleanly while preserving FR
    rows_de_updated = [{**rows[0], "spend_usd": 1500.0}]
    write_parquet_table(
        rows_de_updated,
        part_dir,
        table_name="fct_ad_delivery_weekly",
        partition_cols=["country_iso"],
    )

    df = read_parquet_table(part_dir, table_name="fct_ad_delivery_weekly")
    assert len(df) == 2
    assert float(df.loc[df["country_iso"] == "DE", "spend_usd"].iloc[0]) == 1500.0
    assert float(df.loc[df["country_iso"] == "FR", "spend_usd"].iloc[0]) == 800.0


def test_duckdb_live_override_precedence_and_speed(tmp_path):
    store = LakehouseStore(tmp_path / "lakehouse")

    assets = [
        DimCreativeAsset(
            asset_id="asset_1",
            phash="0011223344556677",
            brand_id_raw="BMW",
            source_connector="meta_ad_lib_eu",
            media_type="image",
            media_uri="https://example.com/1.jpg",
            first_seen_date="2026-01-05",
            last_seen_date="2026-01-12",
        ).model_dump(),
        DimCreativeAsset(
            asset_id="asset_2",
            phash="0011223344556677",  # shared visual pHash with asset_1
            brand_id_raw="BMW",
            source_connector="meta_ad_lib_eu",
            media_type="image",
            media_uri="https://example.com/2.jpg",
            first_seen_date="2026-01-05",
            last_seen_date="2026-01-12",
        ).model_dump(),
    ]
    dissections = [
        DimCreativeDissection(
            media_hash_key="0011223344556677",
            evaluated_brand_id="BMW",
            prompt_version="creative_dissection_v1",
            is_active_version=False,  # inactive old prompt version must be excluded
            model_name="gemini-2.5-flash",
            belongs_to_brand=False,
            matched_brand_id=None,
            confidence=0.4,
            category="OldCat",
            core_concept_summary="old",
            messaging_angle="old",
            visual_style_tags=[],
        ).model_dump(),
        DimCreativeDissection(
            media_hash_key="0011223344556677",
            evaluated_brand_id="BMW",
            prompt_version="creative_dissection_v2",
            is_active_version=True,
            model_name="gemini-2.5-flash",
            belongs_to_brand=False,
            matched_brand_id=None,
            confidence=0.85,
            category="EU Automotive",
            core_concept_summary="Electric sedan",
            messaging_angle="Efficiency",
            visual_style_tags=["studio"],
        ).model_dump(),
    ]
    store.write_table("silver", "dim_creative_asset", assets)
    store.write_table("silver", "dim_creative_dissection", dissections)

    ov_store = HumanOverrideStore(store.overrides_dir)
    t_old = dt.datetime(2026, 1, 10, 10, 0, tzinfo=dt.UTC)
    t_new = dt.datetime(2026, 1, 11, 10, 0, tzinfo=dt.UTC)

    # Older media_hash_key override sets belongs_to_brand=True
    ov_store.write_override(
        target_key_type="media_hash_key",
        target_key_value="0011223344556677",
        field_name="belongs_to_brand",
        human_value="true",
        reviewer="alice",
        created_at=t_old,
    )
    # Newer media_hash_key override sets matched_brand_id=BMW
    ov_store.write_override(
        target_key_type="media_hash_key",
        target_key_value="0011223344556677",
        field_name="matched_brand_id",
        human_value="Audi",
        reviewer="alice",
        created_at=t_old,
    )
    ov_store.write_override(
        target_key_type="media_hash_key",
        target_key_value="0011223344556677",
        field_name="matched_brand_id",
        human_value="BMW",
        reviewer="bob",
        created_at=t_new,
    )
    # Specific asset_id override on asset_1 must beat media_hash_key even if older!
    ov_store.write_override(
        target_key_type="asset_id",
        target_key_value="asset_1",
        field_name="matched_brand_id",
        human_value="Mercedes-Benz",
        reviewer="carol",
        created_at=t_old,
    )

    t0 = time.perf_counter()
    df = store.query_live_dissections()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(df) == 2  # inactive prompt version excluded (no 1:N fanout!)
    r1 = df.loc[df["asset_id"] == "asset_1"].iloc[0]
    r2 = df.loc[df["asset_id"] == "asset_2"].iloc[0]
    assert bool(r1["belongs_to_brand"]) is True
    assert r1["matched_brand_id"] == "Mercedes-Benz"  # asset_id beats media_hash_key
    assert bool(r2["belongs_to_brand"]) is True
    assert r2["matched_brand_id"] == "BMW"  # latest media_hash_key wins over older Audi
    assert elapsed_ms < 500.0  # comfortably fast even with DuckDB startup

    # Verify view query itself after registration is <50ms
    con = duckdb.connect(":memory:")
    register_live_dissection_view(
        con,
        assets_path=store.table_path("silver", "dim_creative_asset"),
        dissection_path=store.table_path("silver", "dim_creative_dissection"),
        overrides_dir=store.overrides_dir,
    )
    q0 = time.perf_counter()
    res = con.execute("SELECT asset_id, matched_brand_id, theme_label FROM vw_live_creative_dissection").fetchdf()
    q_ms = (time.perf_counter() - q0) * 1000.0
    con.close()
    assert isinstance(res, pd.DataFrame) and len(res) == 2
    assert "theme_label" in res.columns
    assert res["theme_label"].isna().all()
    assert q_ms < 50.0


def test_unverified_rejection_and_theme_label_override_precedence(tmp_path):
    store = LakehouseStore(tmp_path / "lakehouse")
    assets = [
        DimCreativeAsset(
            asset_id="asset_rej",
            phash="aabbccdd11223344",
            brand_id_raw="BMW",
            source_connector="meta_ad_lib_eu",
            media_type="image",
            media_uri="https://example.com/rej.jpg",
            first_seen_date="2026-01-05",
            last_seen_date="2026-01-12",
        ).model_dump()
    ]
    dissections = [
        DimCreativeDissection(
            media_hash_key="aabbccdd11223344",
            evaluated_brand_id="BMW",
            prompt_version="creative_dissection_v2",
            is_active_version=True,
            model_name="gemini-2.5-flash",
            belongs_to_brand=True,
            matched_brand_id="BMW",
            confidence=0.9,
            category="EU Automotive",
            core_concept_summary="Sedan",
            messaging_angle="Efficiency",
            visual_style_tags=["studio"],
        ).model_dump()
    ]
    themes = [
        {
            "media_hash_key": "aabbccdd11223344",
            "run_id": "run_1",
            "is_active_run": True,
            "theme_id": "thm_01",
            "theme_label": "EV Range & Charging",
            "theme_scope": "shared_battleground",
            "umap_x": 1.1,
            "umap_y": -0.4,
            "centroid_similarity": 0.92,
        }
    ]
    store.write_table("silver", "dim_creative_asset", assets)
    store.write_table("silver", "dim_creative_dissection", dissections)
    store.write_table("silver", "dim_creative_themes", themes)

    ov_store = HumanOverrideStore(store.overrides_dir)
    t1 = dt.datetime(2026, 2, 1, 10, 0, tzinfo=dt.UTC)
    t2 = dt.datetime(2026, 2, 1, 10, 5, tzinfo=dt.UTC)

    # Prior stale belongs_to_brand=true overridden later by matched_brand_id="Unverified / Other"
    ov_store.write_override(
        target_key_type="asset_id",
        target_key_value="asset_rej",
        field_name="belongs_to_brand",
        human_value="true",
        reviewer="alice",
        created_at=t1,
    )
    ov_store.write_override(
        target_key_type="asset_id",
        target_key_value="asset_rej",
        field_name="matched_brand_id",
        human_value="Unverified / Other",
        reviewer="bob",
        created_at=t2,
    )
    # Global theme_label override via target_key_type='theme_label'
    ov_store.write_override(
        target_key_type="theme_label",
        target_key_value="EV Range & Charging",
        field_name="theme_label",
        human_value="Long-Range EV Performance",
        reviewer="carol",
        created_at=t1,
    )

    df = store.query_live_dissections()
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["belongs_to_brand"]) is False
    assert row["matched_brand_id"] == "Unverified / Other"
    assert row["theme_label"] == "Long-Range EV Performance"
