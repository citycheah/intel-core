"""Idempotent Hive-partitioned Parquet I/O, atomic HITL overrides, and live DuckDB views."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from intel_core.schemas import (
    ARROW_SCHEMAS,
    DimHumanOverride,
    OverrideFieldName,
    OverrideTargetType,
    records_to_arrow,
)


def _atomic_write_bytes(dest: Path, payload: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, dest)


def _clean_df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converts a DataFrame to Pydantic-safe dicts, turning scalar NaN/NaT/pd.NA into None."""
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for k, v in row.items():
            if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
                cleaned[k] = v.tolist()
            elif isinstance(v, (list, dict, tuple, set)):
                cleaned[k] = v
            elif v is None or pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        records.append(cleaned)
    return records


def write_parquet_table(
    table: pa.Table | pd.DataFrame | list[dict[str, Any]],
    dest_path: str | Path,
    *,
    table_name: str | None = None,
    partition_cols: list[str] | None = None,
    overwrite_partitions: bool = True,
) -> Path:
    """Writes a Parquet file or Hive-partitioned directory atomically and idempotently."""
    dest = Path(dest_path)
    if isinstance(table, list):
        if table_name:
            arrow_tbl = records_to_arrow(table_name, table)
        else:
            arrow_tbl = pa.Table.from_pylist(table)
    elif isinstance(table, pd.DataFrame):
        if table_name:
            arrow_tbl = records_to_arrow(table_name, _clean_df_records(table))
        else:
            arrow_tbl = pa.Table.from_pandas(table, preserve_index=False)
    else:
        arrow_tbl = table

    if table_name and table_name in ARROW_SCHEMAS:
        expected_schema = ARROW_SCHEMAS[table_name]
        if arrow_tbl.num_rows == 0:
            arrow_tbl = pa.Table.from_pylist([], schema=expected_schema)
        else:
            arrow_tbl = arrow_tbl.select(expected_schema.names).cast(expected_schema)

    if not partition_cols:
        if dest.suffix.lower() == ".parquet":
            if dest.is_dir():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp.parquet")
            pq.write_table(arrow_tbl, tmp_file, compression="snappy")
            os.replace(tmp_file, dest)
            return dest
        # Directory-based Parquet dataset (compatible with Spark, DuckDB, and PyArrow)
        if dest.is_file():
            dest.unlink()
        staging_dir = dest.parent / f"._{dest.name}_staging_{uuid.uuid4().hex}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            pq.write_table(arrow_tbl, staging_dir / "part-00000.parquet", compression="snappy")
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(staging_dir, dest)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
        return dest

    # Hive-partitioned directory write with dynamic partition swap
    dest.mkdir(parents=True, exist_ok=True)
    staging_dir = dest.parent / f"._{dest.name}_staging_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        if arrow_tbl.num_rows > 0:
            pq.write_to_dataset(
                arrow_tbl,
                root_path=str(staging_dir),
                partition_cols=partition_cols,
                existing_data_behavior="overwrite_or_ignore",
            )
            # Swap leaf partition directories atomically inside dest
            cleared_dirs: set[Path] = set()
            for part_file in staging_dir.rglob("*.parquet"):
                rel_parent = part_file.parent.relative_to(staging_dir)
                target_part_dir = dest / rel_parent
                if overwrite_partitions and target_part_dir not in cleared_dirs:
                    if target_part_dir.exists():
                        shutil.rmtree(target_part_dir)
                    cleared_dirs.add(target_part_dir)
                target_part_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(part_file), str(target_part_dir / part_file.name))
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return dest


def read_parquet_table(
    source_path: str | Path,
    *,
    table_name: str | None = None,
    filters: Any | None = None,
) -> pd.DataFrame:
    """Reads a single Parquet file or Hive-partitioned Parquet directory into a DataFrame."""
    src = Path(source_path)
    if not src.exists():
        if table_name and table_name in ARROW_SCHEMAS:
            return pa.Table.from_pylist([], schema=ARROW_SCHEMAS[table_name]).to_pandas()
        return pd.DataFrame()

    if src.is_dir():
        parquet_files = list(src.rglob("*.parquet"))
        if not parquet_files:
            if table_name and table_name in ARROW_SCHEMAS:
                return pa.Table.from_pylist([], schema=ARROW_SCHEMAS[table_name]).to_pandas()
            return pd.DataFrame()
        dataset = ds.dataset(str(src), format="parquet", partitioning="hive")
        tbl = dataset.to_table(filter=filters)
        return tbl.to_pandas()

    tbl = pq.read_table(src)
    return tbl.to_pandas()


class HumanOverrideStore:
    """Atomic UUID-drop-file store for `dim_human_overrides` (`asset_id` > `media_hash_key`, latest wins)."""

    def __init__(self, overrides_dir: str | Path):
        self.overrides_dir = Path(overrides_dir)
        self.overrides_dir.mkdir(parents=True, exist_ok=True)

    def write_override(
        self,
        *,
        target_key_type: OverrideTargetType,
        target_key_value: str,
        field_name: OverrideFieldName,
        human_value: str,
        reviewer: str,
        ai_value: str | None = None,
        reason: str = "",
        override_id: str | None = None,
        created_at: dt.datetime | None = None,
    ) -> DimHumanOverride:
        record = DimHumanOverride(
            override_id=override_id or str(uuid.uuid4()),
            target_key_type=target_key_type,
            target_key_value=str(target_key_value).strip(),
            field_name=field_name,
            ai_value=ai_value,
            human_value=str(human_value),
            reviewer=str(reviewer).strip() or "analyst",
            reason=str(reason or ""),
            created_at=created_at or dt.datetime.now(dt.UTC),
        )
        dest = self.overrides_dir / f"override_{record.override_id}.json"
        payload = json.dumps(record.model_dump(mode="json"), indent=2).encode("utf-8")
        _atomic_write_bytes(dest, payload)
        return record

    def list_overrides(self) -> list[DimHumanOverride]:
        records: list[DimHumanOverride] = []
        if not self.overrides_dir.exists():
            return records
        for p in sorted(self.overrides_dir.glob("*.json")):
            if p.name.startswith("."):
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                records.append(DimHumanOverride.model_validate(data))
            except Exception as exc:
                raise ValueError(f"Corrupt override drop file {p}: {exc}") from exc
        parquet_files = sorted(self.overrides_dir.glob("*.parquet"))
        for pf in parquet_files:
            tbl = pq.read_table(pf)
            for row in tbl.to_pylist():
                records.append(DimHumanOverride.model_validate(row))
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def to_arrow(self) -> pa.Table:
        records = self.list_overrides()
        return records_to_arrow("dim_human_overrides", records)

    def to_dataframe(self) -> pd.DataFrame:
        return self.to_arrow().to_pandas()


REJECTION_SENTINELS: tuple[str, ...] = (
    "unverified / other",
    "offbrand",
    "none",
    "rejected",
    "false",
)


def is_rejection_value(val: str | None) -> bool:
    """Returns True when an override or AI matched_brand_id explicitly rejects brand attribution."""
    return str(val or "").strip().lower() in REJECTION_SENTINELS


def _parse_bool_str(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "y")


def resolve_brand_override_decision(
    *,
    ai_belongs: bool | None,
    ai_matched: str | None,
    ov_belongs: tuple[str, int, Any] | None = None,
    ov_matched: tuple[str, int, Any] | None = None,
) -> tuple[bool | None, str | None, bool]:
    """Resolves effective `(belongs_to_brand, matched_brand_id, has_brand_override)` using `(key_priority, created_at)`."""
    has_ov = ov_belongs is not None or ov_matched is not None
    matched_wins = False
    if ov_matched is not None:
        if ov_belongs is None:
            matched_wins = True
        else:
            _, b_prio, b_ts = ov_belongs
            _, m_prio, m_ts = ov_matched
            matched_wins = (m_prio < b_prio) or (m_prio == b_prio and m_ts >= b_ts)

    if matched_wins and ov_matched is not None:
        m_val = str(ov_matched[0]).strip()
        if is_rejection_value(m_val):
            return False, m_val, True
        if m_val:
            return True, m_val, True
        return ai_belongs, ai_matched, True

    if ov_belongs is not None:
        belongs = _parse_bool_str(ov_belongs[0])
        matched = str(ov_matched[0]).strip() if (ov_matched is not None and str(ov_matched[0]).strip()) else ai_matched
        return belongs, matched, True

    if is_rejection_value(ai_matched):
        return False, ai_matched, False
    return ai_belongs, ai_matched, has_ov


def register_live_dissection_view(
    con: duckdb.DuckDBPyConnection,
    *,
    assets_path: str | Path,
    dissection_path: str | Path,
    overrides_dir: str | Path | None = None,
    themes_path: str | Path | None = None,
    view_name: str = "vw_live_creative_dissection",
) -> str:
    """Registers a sub-50ms DuckDB view resolving active dissections, optional themes, and `dim_human_overrides`."""
    assets_df = read_parquet_table(assets_path, table_name="dim_creative_asset")
    dissection_df = read_parquet_table(dissection_path, table_name="dim_creative_dissection")
    overrides_df = (
        HumanOverrideStore(overrides_dir).to_dataframe()
        if overrides_dir is not None
        else pd.DataFrame(
            columns=[
                "override_id",
                "target_key_type",
                "target_key_value",
                "field_name",
                "ai_value",
                "human_value",
                "reviewer",
                "reason",
                "created_at",
            ]
        )
    )

    has_themes = False
    if themes_path is not None and Path(themes_path).exists():
        raw_themes = read_parquet_table(themes_path, table_name="dim_creative_themes")
        if not raw_themes.empty and "media_hash_key" in raw_themes.columns:
            active_themes = raw_themes.copy()
            if "is_active_run" in active_themes.columns:
                active_themes = active_themes[active_themes["is_active_run"].astype(bool)]
            sort_cols = [c for c in ("centroid_similarity", "run_id") if c in active_themes.columns]
            if sort_cols:
                active_themes = active_themes.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            active_themes = active_themes.drop_duplicates(subset=["media_hash_key"], keep="first")
            for col in ("theme_id", "theme_label"):
                if col not in active_themes.columns:
                    active_themes[col] = None
            active_themes = active_themes[["media_hash_key", "theme_id", "theme_label"]]
            if not active_themes.empty:
                con.register("_tmp_themes_df", active_themes)
                con.execute("CREATE OR REPLACE TEMP TABLE dim_creative_themes_live AS SELECT * FROM _tmp_themes_df")
                has_themes = True

    con.register("_tmp_assets_df", assets_df)
    con.register("_tmp_dissection_df", dissection_df)
    con.register("_tmp_overrides_df", overrides_df)

    con.execute("CREATE OR REPLACE TEMP TABLE dim_creative_asset_live AS SELECT * FROM _tmp_assets_df")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE dim_creative_dissection_live AS SELECT * FROM _tmp_dissection_df"
    )
    con.execute("CREATE OR REPLACE TEMP TABLE dim_human_overrides_live AS SELECT * FROM _tmp_overrides_df")

    theme_select_sql = (
        "t.theme_id AS ai_theme_id, t.theme_label AS ai_theme_label"
        if has_themes
        else "CAST(NULL AS VARCHAR) AS ai_theme_id, CAST(NULL AS VARCHAR) AS ai_theme_label"
    )
    theme_join_sql = (
        "LEFT JOIN dim_creative_themes_live t ON a.media_hash_key = t.media_hash_key"
        if has_themes
        else ""
    )
    sentinels_sql = ", ".join(f"'{s}'" for s in REJECTION_SENTINELS)

    # Precedence invariant: asset_id (1) > media_hash_key (2) > theme_label (3); within a tier latest created_at wins.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        WITH active_dissection AS (
            SELECT *
            FROM dim_creative_dissection_live
            WHERE is_active_version = TRUE
        ),
        joined AS (
            SELECT
                a.asset_id,
                a.phash,
                a.media_hash_key,
                a.brand_id_raw,
                a.source_connector,
                a.media_type,
                a.media_uri,
                a.aspect_ratio,
                a.duration_sec,
                a.first_seen_date,
                a.last_seen_date,
                a.is_active,
                d.evaluated_brand_id,
                d.prompt_version,
                d.model_name,
                d.belongs_to_brand AS ai_belongs_to_brand,
                d.matched_brand_id AS ai_matched_brand_id,
                d.confidence,
                d.category AS ai_category,
                {theme_select_sql},
                d.core_concept_summary,
                d.messaging_angle,
                d.visual_style_tags,
                d.hook_type,
                d.hook_transcript,
                d.time_to_brand_sec,
                d.has_voiceover,
                d.has_captions,
                d.cta_text,
                d.cta_clarity_score,
                d.format_fit_score
            FROM dim_creative_asset_live a
            LEFT JOIN active_dissection d
                ON a.media_hash_key = d.media_hash_key
            {theme_join_sql}
        ),
        ranked_overrides AS (
            SELECT
                j.asset_id,
                j.evaluated_brand_id,
                o.field_name,
                o.human_value,
                o.reviewer,
                o.created_at,
                CASE
                    WHEN o.target_key_type = 'asset_id' AND o.target_key_value = j.asset_id THEN 1
                    WHEN o.target_key_type = 'media_hash_key' AND o.target_key_value = j.media_hash_key THEN 2
                    WHEN o.target_key_type = 'theme_label'
                        AND NULLIF(TRIM(COALESCE(j.ai_theme_label, '')), '') IS NOT NULL
                        AND LOWER(TRIM(o.target_key_value)) = LOWER(TRIM(j.ai_theme_label)) THEN 3
                    ELSE 99
                END AS key_priority
            FROM joined j
            INNER JOIN dim_human_overrides_live o
                ON (o.target_key_type = 'asset_id' AND o.target_key_value = j.asset_id)
                OR (o.target_key_type = 'media_hash_key' AND o.target_key_value = j.media_hash_key)
                OR (
                    o.target_key_type = 'theme_label'
                    AND NULLIF(TRIM(COALESCE(j.ai_theme_label, '')), '') IS NOT NULL
                    AND LOWER(TRIM(o.target_key_value)) = LOWER(TRIM(j.ai_theme_label))
                )
        ),
        winning_overrides AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY asset_id, evaluated_brand_id, field_name
                        ORDER BY key_priority ASC, created_at DESC
                    ) AS rn
                FROM ranked_overrides
            )
            WHERE rn = 1
        ),
        pivoted AS (
            SELECT
                asset_id,
                evaluated_brand_id,
                MAX(CASE WHEN field_name = 'belongs_to_brand' THEN human_value END) AS ov_belongs_to_brand,
                MIN(CASE WHEN field_name = 'belongs_to_brand' THEN key_priority ELSE 99 END) AS ov_belongs_prio,
                MAX(CASE WHEN field_name = 'belongs_to_brand' THEN created_at END) AS ov_belongs_ts,
                MAX(CASE WHEN field_name = 'matched_brand_id' THEN human_value END) AS ov_matched_brand_id,
                MIN(CASE WHEN field_name = 'matched_brand_id' THEN key_priority ELSE 99 END) AS ov_matched_prio,
                MAX(CASE WHEN field_name = 'matched_brand_id' THEN created_at END) AS ov_matched_ts,
                MAX(CASE WHEN field_name = 'category' THEN human_value END) AS ov_category,
                MAX(CASE WHEN field_name = 'theme_label' THEN human_value END) AS ov_theme_label,
                MAX(reviewer) AS override_reviewer,
                MAX(created_at) AS override_created_at
            FROM winning_overrides
            GROUP BY asset_id, evaluated_brand_id
        )
        SELECT
            j.*,
            j.ai_theme_id AS theme_id,
            CASE
                WHEN p.ov_matched_brand_id IS NOT NULL
                    AND (
                        p.ov_belongs_to_brand IS NULL
                        OR p.ov_matched_prio < p.ov_belongs_prio
                        OR (p.ov_matched_prio = p.ov_belongs_prio AND p.ov_matched_ts >= p.ov_belongs_ts)
                    )
                    THEN CASE
                        WHEN LOWER(TRIM(p.ov_matched_brand_id)) IN ({sentinels_sql}) THEN FALSE
                        WHEN NULLIF(TRIM(p.ov_matched_brand_id), '') IS NOT NULL THEN TRUE
                        ELSE COALESCE(j.ai_belongs_to_brand, FALSE)
                    END
                WHEN p.ov_belongs_to_brand IS NOT NULL
                    THEN LOWER(TRIM(p.ov_belongs_to_brand)) IN ('true', '1', 'yes', 'y')
                WHEN LOWER(TRIM(COALESCE(j.ai_matched_brand_id, ''))) IN ({sentinels_sql})
                    THEN FALSE
                ELSE j.ai_belongs_to_brand
            END AS belongs_to_brand,
            COALESCE(NULLIF(TRIM(p.ov_matched_brand_id), ''), j.ai_matched_brand_id) AS matched_brand_id,
            COALESCE(NULLIF(TRIM(p.ov_category), ''), j.ai_category) AS category,
            COALESCE(NULLIF(TRIM(p.ov_theme_label), ''), j.ai_theme_label) AS theme_label,
            (
                p.ov_belongs_to_brand IS NOT NULL
                OR p.ov_matched_brand_id IS NOT NULL
                OR p.ov_category IS NOT NULL
                OR p.ov_theme_label IS NOT NULL
            ) AS has_human_override,
            p.override_reviewer,
            p.override_created_at
        FROM joined j
        LEFT JOIN pivoted p
            ON j.asset_id = p.asset_id
            AND COALESCE(j.evaluated_brand_id, '') = COALESCE(p.evaluated_brand_id, '')
        """
    )
    return view_name


class LakehouseStore:
    """Convenience accessor over the local Hive/Parquet lakehouse directory layout."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.bronze_dir = self.root_dir / "bronze"
        self.silver_dir = self.root_dir / "silver"
        self.gold_dir = self.root_dir / "gold"
        self.overrides_dir = self.root_dir / "overrides"
        for d in (self.bronze_dir, self.silver_dir, self.gold_dir, self.overrides_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.overrides = HumanOverrideStore(self.overrides_dir)

    def table_path(self, layer: str, table_name: str) -> Path:
        base = {
            "bronze": self.bronze_dir,
            "silver": self.silver_dir,
            "gold": self.gold_dir,
        }.get(layer.lower())
        if base is None:
            raise ValueError(f"Unknown lakehouse layer: {layer!r}")
        return base / table_name

    def write_table(
        self,
        layer: str,
        table_name: str,
        data: pa.Table | pd.DataFrame | list[dict[str, Any]],
        *,
        partition_cols: list[str] | None = None,
    ) -> Path:
        dest = self.table_path(layer, table_name)
        canonical_name = table_name if table_name in ARROW_SCHEMAS else None
        return write_parquet_table(
            data,
            dest,
            table_name=canonical_name,
            partition_cols=partition_cols,
        )

    def read_table(self, layer: str, table_name: str) -> pd.DataFrame:
        src = self.table_path(layer, table_name)
        canonical_name = table_name if table_name in ARROW_SCHEMAS else None
        return read_parquet_table(src, table_name=canonical_name)

    def query_live_dissections(self, *, only_overridden: bool = False) -> pd.DataFrame:
        con = duckdb.connect(":memory:")
        try:
            themes_src = self.table_path("silver", "dim_creative_themes")
            view_name = register_live_dissection_view(
                con,
                assets_path=self.table_path("silver", "dim_creative_asset"),
                dissection_path=self.table_path("silver", "dim_creative_dissection"),
                overrides_dir=self.overrides_dir,
                themes_path=themes_src if themes_src.exists() else None,
            )
            where = "WHERE has_human_override = TRUE" if only_overridden else ""
            return con.execute(f"SELECT * FROM {view_name} {where}").fetchdf()
        finally:
            con.close()
