from PIL import Image, ImageDraw

from intel_core.media import (
    MediaStager,
    build_storyboard_sheet,
    compute_format_fit_score,
    compute_phash_hex,
    phash_distance,
)


def test_phash_stability_and_distance():
    img_a = Image.new("RGB", (64, 64), color=(20, 80, 180))
    draw_a = ImageDraw.Draw(img_a)
    draw_a.rectangle([8, 8, 40, 40], fill=(240, 240, 240))

    img_b = Image.new("RGB", (64, 64), color=(200, 30, 30))
    draw_b = ImageDraw.Draw(img_b)
    draw_b.ellipse([24, 24, 58, 58], fill=(10, 10, 10))

    h_a = compute_phash_hex(img_a)
    h_a_resized = compute_phash_hex(img_a.resize((128, 128)))
    h_b = compute_phash_hex(img_b)

    assert len(h_a) == 16
    assert phash_distance(h_a, h_a_resized) <= 4
    assert phash_distance(h_a, h_b) > 4


def test_media_stager_local_and_download_failed(tmp_path):
    img_path = tmp_path / "hero.jpg"
    Image.new("RGB", (108, 192), color=(15, 90, 200)).save(img_path)

    stager = MediaStager(tmp_path / "staged")
    local_res = stager.stage(str(img_path), media_type="image")
    assert local_res.download_status == "LOCAL"
    assert local_res.phash is not None
    assert local_res.media_hash_key == local_res.phash
    assert local_res.aspect_ratio == "9:16"

    failed_res = stager.stage("https://offline.invalid/video.mp4", media_type="video", allow_network=False)
    assert failed_res.download_status == "DOWNLOAD_FAILED"
    assert failed_res.phash is None
    assert failed_res.media_hash_key == failed_res.asset_id

    sheet = build_storyboard_sheet([img_path, img_path], tmp_path / "sheet.jpg")
    assert sheet is not None and sheet.exists()


def test_format_fit_score_is_deterministic():
    s_good = compute_format_fit_score(
        aspect_ratio="9:16",
        duration_sec=15.0,
        channel_group="tiktok_paid",
        media_type="video",
    )
    s_bad = compute_format_fit_score(
        aspect_ratio="16:9",
        duration_sec=75.0,
        channel_group="tiktok_paid",
        media_type="video",
    )
    assert s_good == 100.0
    assert s_bad < 50.0
