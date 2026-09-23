"""Unit tests for the Gradio industrial inspection UI and inspection callback."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np

from anomavision.ui.gradio_app import build_gradio_interface, run_ui_inspection
from tests.test_api import _build_test_engine


def test_gradio_ui_inspection_and_interface(tmp_path: Path) -> None:
    engine = _build_test_engine(tmp_path)
    demo = build_gradio_interface(engine=engine, bundles_root=engine.bundles_root)
    assert isinstance(demo, gr.Blocks)

    # Test empty input handling
    orig, hmap, ovl, msk, banner, gal, telem = run_ui_inspection(
        engine=engine,
        image=None,
        category="bottle",
        use_sam_refinement=False,
    )
    assert orig is None
    assert "Upload or select" in banner

    # Test anomalous sample inspection
    arr = np.full((64, 64, 3), 80, dtype=np.uint8)
    arr[16:32, 16:32] = 255
    orig, hmap, ovl, msk, banner, gal, telem = run_ui_inspection(
        engine=engine,
        image=arr,
        category="bottle",
        use_sam_refinement=False,
        top_k_similar=3,
    )
    assert orig is not None and orig.shape == (64, 64, 3)
    assert hmap is not None and hmap.shape == (64, 64, 3)
    assert ovl is not None and ovl.shape == (64, 64, 3)
    assert msk is not None
    assert "REJECT — ANOMALY DETECTED" in banner
    assert len(gal) == 3
    parsed_telem = json.loads(telem)
    assert parsed_telem["decision"] == "ANOMALOUS"
    assert parsed_telem["category"] == "bottle"
