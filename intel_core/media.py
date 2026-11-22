"""Media staging, 64-bit perceptual hashing, ffmpeg storyboard extraction, and format-fit scoring."""

from __future__ import annotations

import math
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from intel_core.schemas import (
    canonicalize_media_url,
    compute_asset_id,
    compute_media_hash_key,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def _dct_1d_ortho_matrix(n: int) -> np.ndarray:
    """Builds an orthonormal N x N Type-II DCT matrix without external scipy."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    mat[0, :] *= math.sqrt(1.0 / n)
    mat[1:, :] *= math.sqrt(2.0 / n)
    return mat


_DCT_32 = _dct_1d_ortho_matrix(32)


def compute_phash_hex(pil_img: Image.Image) -> str:
    """Computes a 64-bit DCT perceptual hash (16-char hex string) from a PIL Image."""
    try:
        import imagehash  # type: ignore[import-not-found]

        return str(imagehash.phash(pil_img))
    except ImportError:
        gray = pil_img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
        dct_2d = _DCT_32 @ pixels @ _DCT_32.T
        low_freq = dct_2d[:8, :8].flatten()
        med = float(np.median(low_freq[1:]))
        bits = (low_freq > med).astype(int)
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return f"{val:016x}"


def phash_distance(hash_a: str, hash_b: str) -> int:
    """Computes Hamming distance between two 16-char hex perceptual hashes."""
    return (int(str(hash_a), 16) ^ int(str(hash_b), 16)).bit_count()


def infer_aspect_ratio(width: int, height: int) -> str:
    """Maps raw pixel dimensions to the nearest canonical digital ad aspect ratio."""
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = width / float(height)
    candidates = {
        "9:16": 9.0 / 16.0,
        "4:5": 4.0 / 5.0,
        "1:1": 1.0,
        "16:9": 16.0 / 9.0,
    }
    return min(candidates, key=lambda label: abs(ratio - candidates[label]))


def compute_format_fit_score(
    *,
    aspect_ratio: str | None,
    duration_sec: float | None,
    channel_group: str,
    media_type: Literal["video", "image", "text"] = "image",
) -> float:
    """Deterministic 0-100 creative format-fit score based on channel aspect & duration norms."""
    if media_type == "text":
        return 45.0

    ch = (channel_group or "").strip().lower()
    ar = (aspect_ratio or "").strip()

    # Base aspect-ratio suitability (up to 65 points)
    aspect_table: dict[str, dict[str, float]] = {
        "tiktok_paid": {"9:16": 65.0, "4:5": 45.0, "1:1": 35.0, "16:9": 20.0},
        "meta_paid": {"9:16": 62.0, "4:5": 65.0, "1:1": 60.0, "16:9": 38.0},
        "youtube_paid": {"16:9": 65.0, "9:16": 58.0, "1:1": 42.0, "4:5": 35.0},
        "display": {"1:1": 62.0, "16:9": 60.0, "4:5": 54.0, "9:16": 38.0},
    }
    channel_scores = aspect_table.get(ch, {"9:16": 55.0, "4:5": 55.0, "1:1": 55.0, "16:9": 55.0})
    ar_score = channel_scores.get(ar, 30.0)

    # Duration ergonomics (up to 35 points)
    if media_type == "image" or duration_sec is None:
        dur_score = 30.0 if ch in ("meta_paid", "display") else 20.0
    else:
        dur = float(duration_sec)
        if ch == "tiktok_paid":
            dur_score = 35.0 if 6.0 <= dur <= 25.0 else (26.0 if dur <= 45.0 else 15.0)
        elif ch == "meta_paid":
            dur_score = 35.0 if 6.0 <= dur <= 30.0 else (25.0 if dur <= 60.0 else 16.0)
        elif ch == "youtube_paid":
            dur_score = 35.0 if 10.0 <= dur <= 45.0 else (26.0 if dur <= 90.0 else 18.0)
        else:
            dur_score = 30.0 if dur <= 30.0 else 20.0

    return round(max(0.0, min(100.0, ar_score + dur_score)), 2)


def extract_video_storyboard(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    num_frames: int = 4,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Path]:
    """Extracts up to `num_frames` evenly-spaced JPEG frames via ffmpeg, or [] if unavailable."""
    src = Path(video_path)
    if not src.exists() or shutil.which(ffmpeg_bin) is None:
        return []
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    pattern = target_dir / f"{src.stem}_frame_%02d.jpg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps=1/{max(1, num_frames)}",
        "-frames:v",
        str(num_frames),
        "-q:v",
        "3",
        str(pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return sorted(target_dir.glob(f"{src.stem}_frame_*.jpg"))


def build_storyboard_sheet(frame_paths: list[Path], out_path: str | Path) -> Path | None:
    """Combines 1-4 frame images into a single 2x2 storyboard JPEG for FinOps-efficient Gemini inspection."""
    valid = [p for p in frame_paths if p.exists()]
    if not valid:
        return None
    imgs = [Image.open(p).convert("RGB").resize((320, 320)) for p in valid[:4]]
    cols = 2 if len(imgs) > 1 else 1
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * 320, rows * 320), color=(18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    for idx, im in enumerate(imgs):
        x = (idx % cols) * 320
        y = (idx // cols) * 320
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + 8), f"F{idx + 1}", fill=(255, 220, 90))
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, format="JPEG", quality=88)
    return dest


@dataclass
class StagedMedia:
    """Metadata produced by `MediaStager.stage()` prior to Gemini Vision calls."""

    asset_id: str
    media_uri: str
    media_hash_key: str
    phash: str | None = None
    local_path: Path | None = None
    aspect_ratio: str | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    storyboard_paths: list[Path] = field(default_factory=list)
    download_status: Literal["LOCAL", "DOWNLOADED", "DOWNLOAD_FAILED", "TEXT_ONLY"] = "LOCAL"


class MediaStager:
    """Canonicalizes media URIs, stages local copies, computes 64-bit pHash, and builds hero storyboards."""

    def __init__(self, staging_dir: str | Path):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def stage(
        self,
        media_uri: str,
        *,
        media_type: Literal["video", "image", "text"] = "image",
        aspect_ratio: str | None = None,
        duration_sec: float | None = None,
        allow_network: bool = False,
        timeout_sec: int = 20,
    ) -> StagedMedia:
        canon_uri = canonicalize_media_url(media_uri)
        asset_id = compute_asset_id(canon_uri)

        if media_type == "text":
            return StagedMedia(
                asset_id=asset_id,
                media_uri=canon_uri,
                media_hash_key=compute_media_hash_key(None, asset_id),
                phash=None,
                aspect_ratio=aspect_ratio,
                duration_sec=duration_sec,
                download_status="TEXT_ONLY",
            )

        local_file: Path | None = None
        status: Literal["LOCAL", "DOWNLOADED", "DOWNLOAD_FAILED", "TEXT_ONLY"] = "LOCAL"

        if canon_uri.startswith(("http://", "https://")):
            ext = Path(canon_uri.split("?")[0]).suffix.lower() or (
                ".mp4" if media_type == "video" else ".jpg"
            )
            cached = self.staging_dir / f"{asset_id}{ext}"
            if cached.exists():
                local_file = cached
                status = "DOWNLOADED"
            elif allow_network:
                try:
                    req = urllib.request.Request(canon_uri, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        cached.write_bytes(resp.read())
                    local_file = cached
                    status = "DOWNLOADED"
                except Exception:
                    status = "DOWNLOAD_FAILED"
            else:
                status = "DOWNLOAD_FAILED"
        elif not canon_uri.startswith("gs://"):
            p = Path(media_uri)
            if p.exists() and p.is_file():
                local_file = p
                status = "LOCAL"
            else:
                status = "DOWNLOAD_FAILED"
        else:
            status = "DOWNLOAD_FAILED"

        phash: str | None = None
        width: int | None = None
        height: int | None = None
        detected_ar = aspect_ratio
        storyboard_paths: list[Path] = []

        if local_file is not None and local_file.exists():
            suffix = local_file.suffix.lower()
            if suffix in _IMAGE_EXTS or (media_type == "image" and suffix not in _VIDEO_EXTS):
                try:
                    with Image.open(local_file) as im:
                        width, height = im.size
                        phash = compute_phash_hex(im)
                        if not detected_ar:
                            detected_ar = infer_aspect_ratio(width, height)
                    storyboard_paths = [local_file]
                except Exception:
                    status = "DOWNLOAD_FAILED"
                    local_file = None
            elif suffix in _VIDEO_EXTS or media_type == "video":
                frames_dir = self.staging_dir / "frames" / asset_id
                frames = extract_video_storyboard(local_file, frames_dir, num_frames=4)
                if frames:
                    storyboard_paths = frames
                    with Image.open(frames[0]) as im:
                        width, height = im.size
                        phash = compute_phash_hex(im)
                        if not detected_ar:
                            detected_ar = infer_aspect_ratio(width, height)

        media_hash_key = compute_media_hash_key(phash, asset_id)
        return StagedMedia(
            asset_id=asset_id,
            media_uri=canon_uri,
            media_hash_key=media_hash_key,
            phash=phash,
            local_path=local_file,
            aspect_ratio=detected_ar,
            width=width,
            height=height,
            duration_sec=duration_sec,
            storyboard_paths=storyboard_paths,
            download_status=status,
        )
