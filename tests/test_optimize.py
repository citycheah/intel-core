from intel_core.optimize import simulate_budget_reallocation


def _sample_curves():
    return {
        "brand_id": "BMW",
        "country_iso": "DE",
        "outcome_metric_name": "conversions",
        "is_proxy_outcome": False,
        "is_spend_modeled": True,
        "channels": {
            "own_spend_meta_usd": {
                "baseline_weekly_spend_usd": 40000.0,
                "beta": 500.0,
                "ec50_usd": 25000.0,
                "slope": 1.4,
                "adstock_lambda": 0.35,
            },
            "own_spend_youtube_usd": {
                "baseline_weekly_spend_usd": 15000.0,
                "beta": 650.0,
                "ec50_usd": 20000.0,
                "slope": 1.3,
                "adstock_lambda": 0.45,
            },
            "own_spend_display_usd": {
                "baseline_weekly_spend_usd": 25000.0,
                "beta": 200.0,
                "ec50_usd": 18000.0,
                "slope": 1.2,
                "adstock_lambda": 0.20,
            },
        },
    }


def test_simulate_budget_reallocation_custom_and_optimized():
    curves = _sample_curves()
    opt = simulate_budget_reallocation(curves, total_budget_usd=80000.0, optimize=True)
    assert opt["roi_label"] == "modeled_reach_cpm_proxy"
    assert "Spend is modeled" in opt["disclaimer"]
    assert abs(opt["simulated_total_spend_usd"] - 80000.0) < 1.0
    assert opt["simulated_total_contribution"] >= opt["baseline_total_contribution"]

    # Custom scenario + Google Trends proxy disclaimer check
    proxy_curves = {
        **curves,
        "outcome_metric_name": "google_trends_index",
        "is_proxy_outcome": True,
    }
    sim = simulate_budget_reallocation(
        proxy_curves,
        proposed_spend_by_channel={
            "own_spend_meta_usd": 35000.0,
            "own_spend_youtube_usd": 30000.0,
            "own_spend_display_usd": 15000.0,
        },
    )
    assert sim["roi_label"] == "proxy_index_only"
    assert "NOT financial ROAS" in sim["disclaimer"]
