"""Modular, idempotent pipeline stages for dataset prep, feature extraction, coreset fit, and evaluation."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anomavision.config import CoresetConfig, PipelineConfig
from anomavision.core.coreset import CoresetSelectionResult
from anomavision.core.detector import GlobalCentroidBaseline, PatchCoreDetector
from anomavision.core.drift import DriftReferenceStats
from anomavision.core.retrieval import ReferenceImageSimilarityIndex
from anomavision.datasets.mvtec import (
    MVTecCategoryDataset,
    build_category_manifest,
    download_mvtec_category,
)
from anomavision.evaluation.calibration import fit_unsupervised_calibration
from anomavision.evaluation.metrics import (
    compute_aupro,
    compute_binary_classification_metrics,
    compute_bootstrap_auroc,
    compute_per_defect_auroc,
    compute_pixel_auroc_with_ci,
)
from anomavision.evaluation.visualization import (
    create_ablation_plot,
    create_evaluation_summary_plot,
    create_inspection_panel_figure,
)
from anomavision.modal_storage import VolumeLayout
from anomavision.models.dinov3 import DINOv3FeatureExtractor
from anomavision.models.sam import (
    SAM3ObjectSegmenter,
    SAM31MultiplexAdapter,
    apply_object_mask_refinement,
)
from anomavision.schemas import (
    AblationPoint,
    CalibrationParams,
    CategoryDatasetManifest,
    CategoryEvaluationReport,
)

logger = logging.getLogger(__name__)


def stage_prepare_dataset(
    category: str,
    layout: VolumeLayout,
    cfg: PipelineConfig,
    hf_token: str | None = None,
    force: bool = False,
) -> CategoryDatasetManifest:
    """Stage 1: Download MVTec AD category, validate checksums/masks, and save split manifest."""
    layout.ensure_directories()
    manifest_path = layout.manifests_dir / f"{category}_manifest.json"

    download_mvtec_category(
        category=category,
        target_root=layout.mvtec_root,
        dataset_cfg=cfg.dataset,
        hf_token=hf_token,
        force=force,
    )

    if not force and manifest_path.is_file():
        manifest = CategoryDatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            manifest.seed == cfg.dataset.seed
            and manifest.calibration_fraction == cfg.dataset.calibration_fraction
        ):
            logger.info("Reusing validated manifest for '%s' (%s)", category, manifest.manifest_sha256[:12])
            return manifest

    manifest = build_category_manifest(
        category=category,
        dataset_root=layout.mvtec_root,
        dataset_cfg=cfg.dataset,
        manifest_out_path=manifest_path,
    )
    logger.info(
        "Built manifest for '%s': train_fit=%d, calibration=%d, test_good=%d, test_anomalous=%d",
        category,
        manifest.num_train_fit,
        manifest.num_calibration,
        manifest.num_test_good,
        manifest.num_test_anomalous,
    )
    return manifest


def stage_cache_checkpoints(
    layout: VolumeLayout,
    cfg: PipelineConfig,
    hf_token: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Stage 2: Pre-warm and verify DINOv3, SAM 3, and SAM 3.1 provenance in the persistent Volume."""
    layout.ensure_directories()
    provenance_path = layout.hf_cache_dir / "checkpoint_provenance.json"
    if not force and provenance_path.is_file():
        logger.info("Reusing cached checkpoint provenance from %s", provenance_path)
        return dict(json.loads(provenance_path.read_text(encoding="utf-8")))

    dinov3 = DINOv3FeatureExtractor(cfg.backbone, hf_token=hf_token)
    dinov3_meta = dinov3.metadata()
    del dinov3

    sam_meta: dict[str, Any] = {
        "enabled": cfg.sam.enabled,
        "primary_model_id": cfg.sam.primary_model_id,
        "primary_revision": cfg.sam.primary_revision,
        "primary_weights_sha256": cfg.sam.primary_weights_sha256,
    }
    if cfg.sam.enabled and cfg.sam.adapter_variant == "sam3":
        sam3 = SAM3ObjectSegmenter(cfg.sam, hf_token=hf_token)
        del sam3

    sam31_adapter = SAM31MultiplexAdapter(cfg.sam, hf_token=hf_token)
    sam31_meta = sam31_adapter.inspect_checkpoint_provenance(download_weights=False)

    provenance = {
        "dinov3": dinov3_meta,
        "sam3": sam_meta,
        "sam3_1_optional_adapter": sam31_meta,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return provenance


def stage_extract_embeddings(
    category: str,
    manifest: CategoryDatasetManifest,
    layout: VolumeLayout,
    cfg: PipelineConfig,
    hf_token: str | None = None,
    force: bool = False,
) -> tuple[Path, float, float]:
    """Stage 3: Extract dense DINOv3 patch embeddings and SAM 3 foreground masks.

    Returns:
        Tuple of (embedding_cache_file, mean_inference_latency_ms, peak_gpu_memory_mb).
    """
    fp = cfg.fingerprint()
    emb_dir = layout.embeddings_dir / category / fp
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_file = emb_dir / "dense_embeddings.npz"
    meta_file = emb_dir / "extraction_telemetry.json"

    if not force and emb_file.is_file() and meta_file.is_file():
        telem = json.loads(meta_file.read_text(encoding="utf-8"))
        logger.info("Reusing cached dense embeddings for '%s' at %s", category, emb_file)
        return (
            emb_file,
            float(telem["mean_inference_latency_ms"]),
            float(telem["peak_gpu_memory_mb"]),
        )

    ds = MVTecCategoryDataset(layout.mvtec_root, manifest)
    train_records = ds.get_split("train_fit")
    cal_records = ds.get_split("calibration")
    test_records = ds.get_split("test")

    # Extract unnormalized raw patch features first so both L2-normalized and raw ablations can use them
    raw_backbone_cfg = cfg.backbone.model_copy(update={"l2_normalize": False})
    extractor = DINOv3FeatureExtractor(raw_backbone_cfg, hf_token=hf_token)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_imgs = [ds.load_sample_image(r) for r in train_records]
    cal_imgs = [ds.load_sample_image(r) for r in cal_records]
    test_imgs = [ds.load_sample_image(r) for r in test_records]

    train_batch = extractor.extract_dataset(train_imgs)
    cal_batch = extractor.extract_dataset(cal_imgs)
    test_batch = extractor.extract_dataset(test_imgs)

    # Measure single-image inference latency (mean & p95) on 25 representative test images
    single_latencies_ms: list[float] = []
    _ = extractor.extract_batch(test_imgs[:1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for img in test_imgs[: min(25, len(test_imgs))]:
        t0 = time.perf_counter()
        _ = extractor.extract_batch([img])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_gpu_mb = (
        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if torch.cuda.is_available()
        else 0.0
    )
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract SAM 3 object masks for test images if enabled
    test_sam_masks = np.ones(
        (len(test_imgs), test_imgs[0].shape[0], test_imgs[0].shape[1]),
        dtype=np.uint8,
    )
    if cfg.sam.enabled and cfg.sam.adapter_variant == "sam3":
        sam_segmenter = SAM3ObjectSegmenter(cfg.sam, hf_token=hf_token)
        for idx, img in enumerate(test_imgs):
            mask_bin, _ = sam_segmenter.segment_object(img, category=category)
            test_sam_masks[idx] = mask_bin
        if torch.cuda.is_available():
            peak_gpu_mb = max(
                peak_gpu_mb,
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
            )
        del sam_segmenter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_lat = float(np.mean(single_latencies_ms))
    p95_lat = float(np.percentile(single_latencies_ms, 95))

    np.savez(
        emb_file,
        train_patches_raw=train_batch.patch_embeddings,
        train_global=train_batch.global_embeddings,
        train_sizes=np.asarray(train_batch.original_sizes, dtype=np.int32),
        cal_patches_raw=cal_batch.patch_embeddings,
        cal_global=cal_batch.global_embeddings,
        cal_sizes=np.asarray(cal_batch.original_sizes, dtype=np.int32),
        test_patches_raw=test_batch.patch_embeddings,
        test_global=test_batch.global_embeddings,
        test_sizes=np.asarray(test_batch.original_sizes, dtype=np.int32),
        test_sam_masks=test_sam_masks,
        grid_size=np.asarray(train_batch.grid_size, dtype=np.int32),
        test_latencies_ms=np.asarray(single_latencies_ms, dtype=np.float32),
    )

    telemetry = {
        "category": category,
        "fingerprint": fp,
        "mean_inference_latency_ms": mean_lat,
        "p95_inference_latency_ms": p95_lat,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }
    meta_file.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    return emb_file, mean_lat, peak_gpu_mb


def _maybe_l2_normalize(patches: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return patches.astype(np.float32, copy=False)
    norms = np.linalg.norm(patches, axis=-1, keepdims=True)
    return (patches / np.maximum(norms, 1e-12)).astype(np.float32)


def stage_build_memory_bank(
    category: str,
    manifest: CategoryDatasetManifest,
    emb_file: Path,
    layout: VolumeLayout,
    cfg: PipelineConfig,
    force: bool = False,
    emb_data: Any | None = None,
) -> tuple[PatchCoreDetector, ReferenceImageSimilarityIndex, CoresetSelectionResult]:
    """Stage 4: Construct greedy k-center coreset memory bank and FAISS similarity retrieval index."""
    fp = cfg.fingerprint()
    mb_dir = layout.memory_banks_dir / category / fp
    mb_dir.mkdir(parents=True, exist_ok=True)
    coreset_path = mb_dir / "patchcore_coreset.npz"
    faiss_path = mb_dir / "reference_images.faiss"
    meta_path = mb_dir / "reference_metadata.json"

    if not force and coreset_path.is_file() and faiss_path.is_file() and meta_path.is_file():
        detector = PatchCoreDetector.load(coreset_path, cfg.coreset)
        retrieval_index = ReferenceImageSimilarityIndex.load(faiss_path, meta_path)
        assert detector.coreset_stats is not None
        logger.info("Reusing cached coreset memory bank for '%s' (%d patches)", category, len(detector.coreset_stats.selected_indices))
        return detector, retrieval_index, detector.coreset_stats

    data = emb_data if emb_data is not None else np.load(emb_file)
    train_patches = _maybe_l2_normalize(data["train_patches_raw"], cfg.backbone.l2_normalize)
    train_global = data["train_global"].astype(np.float32)

    detector = PatchCoreDetector(cfg.coreset)
    coreset_res = detector.fit(train_patches)
    detector.save(coreset_path)

    ds = MVTecCategoryDataset(layout.mvtec_root, manifest)
    train_records = ds.get_split("train_fit")

    retrieval_index = ReferenceImageSimilarityIndex(dim=train_global.shape[1])
    retrieval_index.fit(train_global, train_records)
    retrieval_index.save(faiss_path, meta_path)

    logger.info(
        "Built coreset memory bank for '%s': %d / %d patches (%.2f MB)",
        category,
        len(coreset_res.selected_indices),
        coreset_res.total_candidates,
        detector.memory_bank_size_mb,
    )
    return detector, retrieval_index, coreset_res


def stage_calibrate(
    category: str,
    emb_file: Path,
    detector: PatchCoreDetector,
    layout: VolumeLayout,
    cfg: PipelineConfig,
    emb_data: Any | None = None,
) -> tuple[CalibrationParams, DriftReferenceStats, np.ndarray]:
    """Stage 5: Compute unsupervised threshold calibration and MMD drift baseline on held-out normal split."""
    fp = cfg.fingerprint()
    cal_dir = layout.calibration_dir / category / fp
    cal_dir.mkdir(parents=True, exist_ok=True)

    data = emb_data if emb_data is not None else np.load(emb_file)
    train_patches = _maybe_l2_normalize(data["train_patches_raw"], cfg.backbone.l2_normalize)
    cal_patches = _maybe_l2_normalize(data["cal_patches_raw"], cfg.backbone.l2_normalize)
    cal_sizes = [tuple(int(x) for x in row) for row in data["cal_sizes"]]
    grid_size = (int(data["grid_size"][0]), int(data["grid_size"][1]))

    cal_pred = detector.predict_batch(
        patch_embeddings=cal_patches,
        grid_size=grid_size,
        original_sizes=cal_sizes,  # type: ignore[arg-type]
    )

    n_tr, n_p, dim = train_patches.shape
    drift_stats = DriftReferenceStats.from_reference_patches(
        train_patches=train_patches.reshape(n_tr * n_p, dim),
        calibration_patches_by_image=cal_patches,
        seed=cfg.dataset.seed,
    )

    cal_params = fit_unsupervised_calibration(
        category=category,
        calibration_image_scores=cal_pred.image_scores,
        calibration_anomaly_maps=cal_pred.anomaly_maps,
        reference_centroid_norm=float(np.linalg.norm(np.mean(train_patches, axis=(0, 1)))),
        reference_mmd_baseline=drift_stats.baseline_mmd_mean,
    )

    (cal_dir / "calibration.json").write_text(cal_params.model_dump_json(indent=2), encoding="utf-8")
    np.savez_compressed(
        cal_dir / "drift_reference.npz",
        reference_anchors=drift_stats.reference_anchors,
        baseline_mmd_mean=np.float32(drift_stats.baseline_mmd_mean),
        baseline_mmd_std=np.float32(drift_stats.baseline_mmd_std),
        alert_threshold=np.float32(drift_stats.alert_threshold),
    )
    return cal_params, drift_stats, cal_pred.image_scores


def stage_evaluate_and_ablate(
    category: str,
    manifest: CategoryDatasetManifest,
    emb_file: Path,
    detector: PatchCoreDetector,
    retrieval_index: ReferenceImageSimilarityIndex,
    coreset_res: CoresetSelectionResult,
    cal_params: CalibrationParams,
    cal_scores: np.ndarray,
    layout: VolumeLayout,
    cfg: PipelineConfig,
    run_ablations: bool = True,
    emb_data: Any | None = None,
) -> tuple[CategoryEvaluationReport, list[AblationPoint]]:
    """Stage 6: Evaluate detector on held-out test set, compare against baseline, and run ablations."""
    fp = cfg.fingerprint()
    data = emb_data if emb_data is not None else np.load(emb_file)
    meta_file = emb_file.parent / "extraction_telemetry.json"
    telem = json.loads(meta_file.read_text(encoding="utf-8"))

    train_patches = _maybe_l2_normalize(data["train_patches_raw"], cfg.backbone.l2_normalize)
    train_global = data["train_global"].astype(np.float32)
    test_patches = _maybe_l2_normalize(data["test_patches_raw"], cfg.backbone.l2_normalize)
    test_global = data["test_global"].astype(np.float32)
    test_sizes = [tuple(int(x) for x in row) for row in data["test_sizes"]]
    test_sam_masks = data["test_sam_masks"]
    grid_size = (int(data["grid_size"][0]), int(data["grid_size"][1]))

    ds = MVTecCategoryDataset(layout.mvtec_root, manifest)
    test_records = ds.get_split("test")
    test_labels = np.array([1 if r.is_anomalous else 0 for r in test_records], dtype=np.int32)
    test_defects = [r.defect_type for r in test_records]
    gt_masks = [ds.load_sample_mask(r) for r in test_records]

    # Measure patch k-NN scoring + image retrieval latency
    t_knn_start = time.perf_counter()
    test_pred = detector.predict_batch(
        patch_embeddings=test_patches,
        grid_size=grid_size,
        original_sizes=test_sizes,  # type: ignore[arg-type]
    )
    knn_ms_per_img = ((time.perf_counter() - t_knn_start) * 1000.0) / max(1, len(test_records))

    ret_latencies: list[float] = []
    for g_emb in test_global:
        _, r_lat = retrieval_index.query(g_emb, top_k=4)
        ret_latencies.append(r_lat)
    mean_ret_ms = float(np.mean(ret_latencies)) + knn_ms_per_img

    # Apply SAM 3 foreground mask refinement to pixel-level anomaly maps when enabled
    refined_maps: list[np.ndarray] = []
    for i, amap in enumerate(test_pred.anomaly_maps):
        if cfg.sam.enabled and cfg.sam.adapter_variant != "disabled":
            refined_maps.append(
                apply_object_mask_refinement(
                    anomaly_map=amap,
                    object_mask=test_sam_masks[i],
                    background_suppression_factor=cfg.sam.background_suppression_factor,
                )
            )
        else:
            refined_maps.append(amap)

    # Fit and evaluate lightweight Global Centroid Baseline on the exact same splits
    baseline = GlobalCentroidBaseline()
    baseline.fit(train_patches, train_global)
    base_scores, base_maps = baseline.predict_batch(
        patch_embeddings=test_patches,
        global_embeddings=test_global,
        grid_size=grid_size,
        original_sizes=test_sizes,  # type: ignore[arg-type]
        gaussian_sigma=cfg.coreset.gaussian_sigma,
    )
    base_img_auroc = compute_bootstrap_auroc(test_labels, base_scores, n_bootstrap=200, seed=cfg.dataset.seed)
    base_pix_auroc = compute_pixel_auroc_with_ci(gt_masks, base_maps, n_bootstrap=30, seed=cfg.dataset.seed)

    # Primary detector metrics
    img_auroc = compute_bootstrap_auroc(test_labels, test_pred.image_scores, n_bootstrap=1000, seed=cfg.dataset.seed)
    pix_auroc = compute_pixel_auroc_with_ci(gt_masks, refined_maps, n_bootstrap=100, seed=cfg.dataset.seed)
    aupro_val = compute_aupro(gt_masks, refined_maps)
    bin_metrics = compute_binary_classification_metrics(
        labels=test_labels,
        scores=test_pred.image_scores,
        threshold=cal_params.image_threshold,
    )
    per_defect = compute_per_defect_auroc(test_defects, test_pred.image_scores)

    total_inf_mean = float(telem["mean_inference_latency_ms"]) + knn_ms_per_img
    total_inf_p95 = float(telem["p95_inference_latency_ms"]) + knn_ms_per_img

    report = CategoryEvaluationReport(
        category=category,
        config_fingerprint=fp,
        backbone_model_id=cfg.backbone.model_id,
        backbone_revision=cfg.backbone.revision,
        sam_variant=cfg.sam.adapter_variant if cfg.sam.enabled else "disabled",
        image_auroc=img_auroc,
        pixel_auroc=pix_auroc,
        aupro=aupro_val,
        precision_at_threshold=bin_metrics["precision"],
        recall_at_threshold=bin_metrics["recall"],
        f1_at_threshold=bin_metrics["f1"],
        f1_oracle_upper_bound=bin_metrics["f1_oracle_upper_bound"],
        threshold_used=cal_params.image_threshold,
        threshold_procedure="unsupervised_calibration_quantile_0.98_plus_gaussian_tail",
        baseline_image_auroc=base_img_auroc.value,
        baseline_pixel_auroc=base_pix_auroc.value,
        baseline_name="GlobalCentroidMahalanobis+SpatialTemplate",
        memory_bank_patches=len(coreset_res.selected_indices),
        total_train_patches=coreset_res.total_candidates,
        coreset_ratio_actual=coreset_res.coreset_ratio_actual,
        memory_bank_size_mb=round(detector.memory_bank_size_mb, 4),
        inference_latency_ms_mean=round(total_inf_mean, 3),
        inference_latency_ms_p95=round(total_inf_p95, 3),
        retrieval_latency_ms_mean=round(mean_ret_ms, 3),
        peak_gpu_memory_mb=round(float(telem["peak_gpu_memory_mb"]), 2),
        per_defect_auroc=per_defect,
    )

    # Save report and generate visualizations
    exp_dir = layout.experiments_dir / category
    vis_dir = layout.visualizations_dir / category
    exp_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "evaluation_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    create_evaluation_summary_plot(
        category=category,
        test_labels=test_labels,
        test_scores=test_pred.image_scores,
        baseline_scores=base_scores,
        calibration_scores=cal_scores,
        threshold=cal_params.image_threshold,
        report=report,
        out_path=vis_dir / f"evaluation_summary_{category}.png",
    )

    # Generate sample qualitative panels across normal, defect types, and boundary/hardest case
    _render_representative_panels(
        category=category,
        ds=ds,
        test_records=test_records,
        gt_masks=gt_masks,
        refined_maps=refined_maps,
        test_sam_masks=test_sam_masks,
        test_scores=test_pred.image_scores,
        cal_params=cal_params,
        vis_dir=vis_dir,
    )

    # Generate a downsampled (224x224) copy of gt_masks for fast ablation Pixel AUROC & AUPRO
    from anomavision.datasets.mvtec import load_mask_binary

    ablation_sizes = [(224, 224) for _ in test_records]
    ablation_gt_masks = [
        load_mask_binary(layout.mvtec_root / r.mask_relpath, target_shape=(224, 224))
        if r.mask_relpath is not None
        else np.zeros((224, 224), dtype=np.uint8)
        for r in test_records
    ]

    ablation_points: list[AblationPoint] = []
    if run_ablations:
        ablation_points = _run_mathematical_ablations(
            category=category,
            train_patches_raw=data["train_patches_raw"],
            test_patches_raw=data["test_patches_raw"],
            grid_size=grid_size,
            test_sizes=ablation_sizes,
            test_labels=test_labels,
            gt_masks=ablation_gt_masks,
            backbone_latency_ms=float(telem["mean_inference_latency_ms"]),
            cfg=cfg,
        )
        (exp_dir / "ablation_results.json").write_text(
            json.dumps([p.model_dump(mode="json") for p in ablation_points], indent=2),
            encoding="utf-8",
        )
        create_ablation_plot(ablation_points, out_path=vis_dir / f"ablation_{category}.png")

    return report, ablation_points


def _render_representative_panels(
    category: str,
    ds: MVTecCategoryDataset,
    test_records: list[Any],
    gt_masks: list[np.ndarray],
    refined_maps: list[np.ndarray],
    test_sam_masks: np.ndarray,
    test_scores: np.ndarray,
    cal_params: CalibrationParams,
    vis_dir: Path,
) -> None:
    """Save qualitative inspection panels for each defect type plus the hardest boundary sample."""
    seen_defects: set[str] = set()
    global_vmin = float(cal_params.cal_score_min * 0.7)
    global_vmax = float(max(np.percentile([ np.max(m) for m in refined_maps ], 90), cal_params.pixel_threshold * 1.3))

    for i, rec in enumerate(test_records):
        if rec.defect_type not in seen_defects:
            seen_defects.add(rec.defect_type)
            img_rgb = ds.load_sample_image(rec)
            decision = "ANOMALOUS" if test_scores[i] >= cal_params.image_threshold else "NORMAL"
            title = (
                f"{category.upper()} | Ground Truth: {rec.defect_type} | "
                f"Score: {test_scores[i]:.4f} (Thresh: {cal_params.image_threshold:.4f}) -> {decision}"
            )
            create_inspection_panel_figure(
                image_rgb=img_rgb,
                gt_mask=gt_masks[i],
                anomaly_map=refined_maps[i],
                sam_mask=test_sam_masks[i],
                pixel_threshold=cal_params.pixel_threshold,
                title=title,
                out_path=vis_dir / f"sample_{category}_{rec.defect_type}.png",
                vmin=global_vmin,
                vmax=global_vmax,
            )

    # Also find the hardest/closest-to-threshold sample for honest failure/boundary analysis
    margins = np.abs(test_scores - cal_params.image_threshold)
    hardest_idx = int(np.argmin(margins))
    hard_rec = test_records[hardest_idx]
    img_rgb = ds.load_sample_image(hard_rec)
    decision = "ANOMALOUS" if test_scores[hardest_idx] >= cal_params.image_threshold else "NORMAL"
    title = (
        f"Boundary / Hardest Case: {hard_rec.sample_id} (GT: {hard_rec.defect_type}) | "
        f"Score: {test_scores[hardest_idx]:.4f} vs Thresh: {cal_params.image_threshold:.4f} ({decision})"
    )
    create_inspection_panel_figure(
        image_rgb=img_rgb,
        gt_mask=gt_masks[hardest_idx],
        anomaly_map=refined_maps[hardest_idx],
        sam_mask=test_sam_masks[hardest_idx],
        pixel_threshold=cal_params.pixel_threshold,
        title=title,
        out_path=vis_dir / f"boundary_case_{category}.png",
        vmin=global_vmin,
        vmax=global_vmax,
    )


def _run_mathematical_ablations(
    category: str,
    train_patches_raw: np.ndarray,
    test_patches_raw: np.ndarray,
    grid_size: tuple[int, int],
    test_sizes: list[tuple[int, int]],
    test_labels: np.ndarray,
    gt_masks: list[np.ndarray],
    backbone_latency_ms: float,
    cfg: PipelineConfig,
) -> list[AblationPoint]:
    """Compare L2-normalized vs Raw embeddings, Cosine vs Euclidean distance, and Coreset ratios.

    Exploits the prefix-nesting property of deterministic greedy k-center (with fixed seed and
    JL projection): the first K_1 centers of a greedy k-center sequence of length K_2 > K_1 are
    mathematically identical to running greedy k-center to length K_1. Thus we compute one master
    k-center sequence for L2-normalized patches and one for raw patches, then slice exact prefixes.
    """
    n_tr, n_p, dim = train_patches_raw.shape
    total_m = n_tr * n_p
    tr_l2 = _maybe_l2_normalize(train_patches_raw, True).reshape(total_m, dim)
    te_l2 = _maybe_l2_normalize(test_patches_raw, True)
    tr_raw = _maybe_l2_normalize(train_patches_raw, False).reshape(total_m, dim)
    te_raw = _maybe_l2_normalize(test_patches_raw, False)

    # Cap master ablation coreset at 8192 patches so GPU k-center completes in <2 seconds
    ablation_max_k = min(cfg.coreset.max_coreset_size, 8192)
    master_l2_cfg = cfg.coreset.model_copy(update={"coreset_ratio": 0.25, "max_coreset_size": ablation_max_k})
    master_l2_det = PatchCoreDetector(master_l2_cfg)
    master_l2_res = master_l2_det.fit(tr_l2)

    master_raw_cfg = cfg.coreset.model_copy(update={"coreset_ratio": 0.10, "max_coreset_size": ablation_max_k})
    master_raw_det = PatchCoreDetector(master_raw_cfg)
    master_raw_res = master_raw_det.fit(tr_raw)

    configs_to_test: list[tuple[str, bool, str, float]] = [
        ("cos_l2_r02", True, "cosine", 0.02),
        ("cos_l2_r05", True, "cosine", 0.05),
        ("cos_l2_r10", True, "cosine", 0.10),
        ("cos_l2_r25", True, "cosine", 0.25),
        ("euc_l2_r10", True, "euclidean", 0.10),
        ("euc_raw_r10", False, "euclidean", 0.10),
        ("cos_raw_r10", False, "cosine", 0.10),
    ]

    results: list[AblationPoint] = []
    for exp_id, l2_norm, metric, ratio in configs_to_test:
        te_p = te_l2 if l2_norm else te_raw
        master_res = master_l2_res if l2_norm else master_raw_res
        target_k = max(1, min(int(round(total_m * ratio)), len(master_res.selected_indices)))
        prefix_emb = master_res.coreset_embeddings[:target_k]

        ab_cfg = CoresetConfig(
            coreset_ratio=ratio,
            max_coreset_size=ablation_max_k,
            projection_dim=cfg.coreset.projection_dim,
            metric=metric,  # type: ignore[arg-type]
            n_neighbors=cfg.coreset.n_neighbors,
            score_top_k_patches=cfg.coreset.score_top_k_patches,
            gaussian_sigma=cfg.coreset.gaussian_sigma,
            seed=cfg.coreset.seed,
        )
        det = PatchCoreDetector(ab_cfg)
        det._set_memory_bank(prefix_emb)

        t0 = time.perf_counter()
        pred = det.predict_batch(te_p, grid_size=grid_size, original_sizes=test_sizes)
        ret_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(test_labels))

        i_auroc = compute_bootstrap_auroc(test_labels, pred.image_scores, n_bootstrap=50, seed=cfg.dataset.seed)
        p_auroc = compute_pixel_auroc_with_ci(gt_masks, pred.anomaly_maps, n_bootstrap=10, seed=cfg.dataset.seed)
        aupro = compute_aupro(gt_masks, pred.anomaly_maps, num_thresholds=60)

        results.append(
            AblationPoint(
                experiment_id=f"{category}_{exp_id}",
                category=category,
                l2_normalize=l2_norm,
                metric=metric,  # type: ignore[arg-type]
                coreset_ratio=ratio,
                memory_bank_patches=target_k,
                memory_bank_size_mb=round(det.memory_bank_size_mb, 4),
                image_auroc=round(i_auroc.value, 5),
                pixel_auroc=round(p_auroc.value, 5),
                aupro=round(aupro, 5),
                inference_latency_ms=round(backbone_latency_ms + ret_ms, 3),
                retrieval_latency_ms=round(ret_ms, 3),
            )
        )
    return results


def stage_package_bundle(
    category: str,
    manifest: CategoryDatasetManifest,
    layout: VolumeLayout,
    cfg: PipelineConfig,
) -> Path:
    """Stage 7: Package all inference artifacts and sample reference images into a portable bundle."""
    fp = cfg.fingerprint()
    bundle_dir = layout.bundles_dir / category
    bundle_dir.mkdir(parents=True, exist_ok=True)

    mb_dir = layout.memory_banks_dir / category / fp
    cal_dir = layout.calibration_dir / category / fp
    exp_dir = layout.experiments_dir / category
    vis_dir = layout.visualizations_dir / category

    shutil.copy2(mb_dir / "patchcore_coreset.npz", bundle_dir / "patchcore_coreset.npz")
    shutil.copy2(mb_dir / "reference_images.faiss", bundle_dir / "reference_images.faiss")
    shutil.copy2(mb_dir / "reference_metadata.json", bundle_dir / "reference_metadata.json")
    shutil.copy2(cal_dir / "calibration.json", bundle_dir / "calibration.json")
    shutil.copy2(cal_dir / "drift_reference.npz", bundle_dir / "drift_reference.npz")
    shutil.copy2(exp_dir / "evaluation_report.json", bundle_dir / "evaluation_report.json")
    if (exp_dir / "ablation_results.json").is_file():
        shutil.copy2(exp_dir / "ablation_results.json", bundle_dir / "ablation_results.json")

    cfg.to_yaml(bundle_dir / "pipeline_config.yaml")

    # Copy all visualization plots
    plots_dir = bundle_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for png in vis_dir.glob("*.png"):
        shutil.copy2(png, plots_dir / png.name)

    # Copy a curated set of real normal reference images and real test examples so local UI/API
    # can display authentic reference thumbnails and interactive test examples without downloading 5GB
    refs_dir = bundle_dir / "reference_thumbnails"
    examples_dir = bundle_dir / "examples"
    refs_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    ds = MVTecCategoryDataset(layout.mvtec_root, manifest)
    for rec in ds.get_split("train_fit")[:25]:
        src_p = layout.mvtec_root / rec.image_relpath
        dst_p = refs_dir / Path(rec.image_relpath).name
        shutil.copy2(src_p, dst_p)

    seen_def: dict[str, int] = {}
    for rec in ds.get_split("test"):
        count = seen_def.get(rec.defect_type, 0)
        if count < 2:
            seen_def[rec.defect_type] = count + 1
            src_p = layout.mvtec_root / rec.image_relpath
            dst_p = examples_dir / f"{rec.defect_type}_{Path(rec.image_relpath).name}"
            shutil.copy2(src_p, dst_p)

    return bundle_dir


def run_category_pipeline(
    category: str,
    layout: VolumeLayout,
    cfg: PipelineConfig,
    hf_token: str | None = None,
    force: bool = False,
    run_ablations: bool = True,
    on_checkpoint_cb: Any | None = None,
) -> tuple[CategoryEvaluationReport, list[AblationPoint], Path]:
    """Execute stages 1 through 7 for a single MVTec AD category."""
    bundle_dir = layout.bundles_dir / category
    rep_file = bundle_dir / "evaluation_report.json"
    ab_file = bundle_dir / "ablation_results.json"
    if not force and rep_file.is_file() and (not run_ablations or ab_file.is_file()):
        cached_rep = CategoryEvaluationReport.model_validate_json(rep_file.read_text(encoding="utf-8"))
        if cached_rep.config_fingerprint == cfg.fingerprint():
            cached_ab: list[AblationPoint] = []
            if ab_file.is_file():
                raw_ab = json.loads(ab_file.read_text(encoding="utf-8"))
                cached_ab = [AblationPoint.model_validate(item) for item in raw_ab]
            logger.info("Reusing completed bundle for category '%s'", category)
            return cached_rep, cached_ab, bundle_dir

    manifest = stage_prepare_dataset(category, layout, cfg, hf_token=hf_token, force=force)
    if on_checkpoint_cb is not None:
        on_checkpoint_cb()
    emb_file, _, _ = stage_extract_embeddings(
        category, manifest, layout, cfg, hf_token=hf_token, force=force
    )
    if on_checkpoint_cb is not None:
        on_checkpoint_cb()

    # Load embedding archive once in memory and share across stages 4, 5, and 6
    raw_npz = np.load(emb_file)
    emb_data = {k: raw_npz[k] for k in raw_npz.files}

    detector, retrieval_index, coreset_res = stage_build_memory_bank(
        category, manifest, emb_file, layout, cfg, force=force, emb_data=emb_data
    )
    if on_checkpoint_cb is not None:
        on_checkpoint_cb()
    cal_params, _, cal_scores = stage_calibrate(
        category, emb_file, detector, layout, cfg, emb_data=emb_data
    )
    report, ablations = stage_evaluate_and_ablate(
        category=category,
        manifest=manifest,
        emb_file=emb_file,
        detector=detector,
        retrieval_index=retrieval_index,
        coreset_res=coreset_res,
        cal_params=cal_params,
        cal_scores=cal_scores,
        layout=layout,
        cfg=cfg,
        run_ablations=run_ablations,
        emb_data=emb_data,
    )
    bundle_dir = stage_package_bundle(category, manifest, layout, cfg)
    if on_checkpoint_cb is not None:
        on_checkpoint_cb()
    return report, ablations, bundle_dir
