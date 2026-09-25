"""End-to-end validation against real MVTec AD evaluation artifacts, FastAPI, and Gradio UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anomavision.api.app import create_app
from anomavision.pipeline.inference import InferenceEngine
from anomavision.schemas import AblationPoint, CategoryEvaluationReport
from anomavision.ui.gradio_app import run_ui_inspection


def test_real_evaluation_artifacts_and_live_inference() -> None:
    summary_path = Path("artifacts/evaluation_summary.json")
    bundles_root = Path("artifacts/bundles")
    if not summary_path.is_file() or not (bundles_root / "bottle" / "patchcore_coreset.npz").is_file():
        pytest.skip("Real MVTec AD bundles not present in CI environment")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reports = [CategoryEvaluationReport.model_validate(r) for r in summary["reports"]]
    ablations = [AblationPoint.model_validate(a) for a in summary["ablations"]]

    assert len(reports) == 2
    by_cat = {r.category: r for r in reports}
    assert by_cat["bottle"].image_auroc.value >= 0.99
    assert by_cat["metal_nut"].image_auroc.value >= 0.99
    assert len(ablations) == 14

    # Instantiate real InferenceEngine with real DINOv3 backbone and real MVTec AD bundles
    engine = InferenceEngine(bundles_root=bundles_root, load_sam=False, device="cpu")
    assert set(engine.available_categories) == {"bottle", "metal_nut"}

    app = create_app(engine=engine)
    client = TestClient(app)

    bottle_good = bundles_root / "bottle" / "examples" / "good_000.png"
    bottle_broken = bundles_root / "bottle" / "examples" / "broken_large_000.png"

    with bottle_good.open("rb") as f:
        r_good = client.post(
            "/predict",
            files={"file": ("good_000.png", f.read(), "image/png")},
            data={"category": "bottle", "top_k_similar": "3"},
        )
    assert r_good.status_code == 200
    assert r_good.json()["decision_label"] == "NORMAL"

    with bottle_broken.open("rb") as f:
        r_broken = client.post(
            "/predict",
            files={"file": ("broken_large_000.png", f.read(), "image/png")},
            data={"category": "bottle", "top_k_similar": "3"},
        )
    assert r_broken.status_code == 200
    assert r_broken.json()["decision_label"] == "ANOMALOUS"
    assert r_broken.json()["anomaly_score"] > r_good.json()["anomaly_score"] * 3.0

    # Also validate Gradio UI callback on a real metal_nut scratch example
    from PIL import Image

    nut_scratch = bundles_root / "metal_nut" / "examples" / "scratch_000.png"
    with Image.open(nut_scratch) as pil_nut:
        orig, hmap, ovl, msk, banner, gal, telem = run_ui_inspection(
            engine=engine,
            image=pil_nut,
            category="metal_nut",
            use_sam_refinement=False,
            top_k_similar=4,
        )
    assert orig is not None
    assert hmap is not None
    assert ovl is not None
    assert "REJECT — ANOMALY DETECTED" in banner
    assert len(gal) == 4
