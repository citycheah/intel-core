from intel_core.evals import (
    make_golden_stub_transport,
    run_golden_vision_eval,
)
from intel_core.gemini import FinOpsGuard, GeminiMediaClient


def test_golden_vision_eval_50_asset_benchmark(tmp_path):
    ledger = tmp_path / "eval_costs.csv"
    client = GeminiMediaClient(
        prompt_version="creative_dissection_v2",
        transport=make_golden_stub_transport(),
        finops_guard=FinOpsGuard(max_budget_usd=5.0, ledger_path=ledger),
    )
    report = run_golden_vision_eval(client)
    assert report["total_cases"] == 50
    assert report["brand_f1"] >= 0.99
    assert report["false_positive_rate"] == 0.0
    assert report["hook_accuracy"] == 1.0
    assert report["passed"] is True
    assert report["cost_per_1k_assets_usd"] > 0.0
