"""MVTec AD dataset acquisition, integrity verification, and deterministic anti-leakage splitting."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from anomavision.config import DatasetConfig
from anomavision.schemas import CategoryDatasetManifest, SampleRecord

logger = logging.getLogger(__name__)

VALID_MVTEC_CATEGORIES: frozenset[str] = frozenset(
    {
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    }
)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image file and convert deterministically to uint8 RGB array (H, W, 3)."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def load_mask_binary(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a ground-truth anomaly mask as binary uint8 array (H, W) in {0, 1}."""
    with Image.open(path) as img:
        gray = img.convert("L")
        if target_shape is not None and (gray.height, gray.width) != target_shape:
            gray = gray.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST)
        arr = np.asarray(gray, dtype=np.uint8)
        return (arr > 127).astype(np.uint8)


def download_mvtec_category(
    category: str,
    target_root: Path,
    dataset_cfg: DatasetConfig,
    hf_token: str | None = None,
    force: bool = False,
) -> Path:
    """Download an authentic MVTec AD category into target_root/<category> if not already cached."""
    if category not in VALID_MVTEC_CATEGORIES:
        raise ValueError(f"Unsupported MVTec AD category '{category}'. Valid: {sorted(VALID_MVTEC_CATEGORIES)}")

    category_dir = target_root / category
    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    if not force and train_good_dir.is_dir() and test_dir.is_dir() and gt_dir.is_dir():
        existing_train = list(train_good_dir.glob("*.png"))
        if len(existing_train) > 0:
            logger.info("Reusing cached MVTec AD category '%s' at %s (%d train samples)", category, category_dir, len(existing_train))
            return category_dir

    target_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading MVTec AD category '%s' from %s (rev %s) to %s",
        category,
        dataset_cfg.hf_dataset_id,
        dataset_cfg.hf_revision[:10],
        target_root,
    )
    snapshot_download(
        repo_id=dataset_cfg.hf_dataset_id,
        repo_type="dataset",
        revision=dataset_cfg.hf_revision,
        local_dir=str(target_root),
        allow_patterns=[
            f"{category}/**",
            "license.txt",
            "readme.txt",
        ],
        token=hf_token,
    )

    if not (train_good_dir.is_dir() and test_dir.is_dir() and gt_dir.is_dir()):
        raise RuntimeError(f"Downloaded MVTec AD category '{category}' is missing required subdirectories in {category_dir}")

    return category_dir


def build_category_manifest(
    category: str,
    dataset_root: Path,
    dataset_cfg: DatasetConfig,
    manifest_out_path: Path | None = None,
) -> CategoryDatasetManifest:
    """Validate all images and masks for a category and construct a deterministic split manifest."""
    category_dir = dataset_root / category
    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    if not train_good_dir.is_dir():
        raise FileNotFoundError(f"Missing train/good directory: {train_good_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing ground_truth directory: {gt_dir}")

    train_files = sorted(train_good_dir.glob("*.png"))
    if len(train_files) < 4:
        raise ValueError(f"Category '{category}' has insufficient training images ({len(train_files)})")

    # Deterministic shuffling of sorted training files for train_fit vs calibration split
    rng = random.Random(dataset_cfg.seed)
    indices = list(range(len(train_files)))
    rng.shuffle(indices)
    num_cal = max(1, int(round(len(train_files) * dataset_cfg.calibration_fraction)))
    cal_index_set = set(indices[:num_cal])

    samples: list[SampleRecord] = []
    for idx, img_path in enumerate(train_files):
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            width, height = img.size
        split_name: str = "calibration" if idx in cal_index_set else "train_fit"
        rel_path = img_path.relative_to(dataset_root).as_posix()
        samples.append(
            SampleRecord(
                sample_id=f"{category}/train/good/{img_path.stem}",
                category=category,
                split=split_name,  # type: ignore[arg-type]
                defect_type="good",
                is_anomalous=False,
                image_relpath=rel_path,
                mask_relpath=None,
                image_sha256=compute_file_sha256(img_path),
                mask_sha256=None,
                width=width,
                height=height,
            )
        )

    defect_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    defect_types: list[str] = []
    num_test_good = 0
    num_test_anomalous = 0

    for defect_dir in defect_dirs:
        defect_name = defect_dir.name
        defect_types.append(defect_name)
        is_anom = defect_name != "good"
        test_images = sorted(defect_dir.glob("*.png"))
        if not test_images:
            raise ValueError(f"Empty test defect directory: {defect_dir}")

        for img_path in test_images:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                width, height = img.size

            mask_relpath: str | None = None
            mask_sha256: str | None = None

            if is_anom:
                mask_path = gt_dir / defect_name / f"{img_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Missing ground-truth mask {mask_path} for anomalous test image {img_path}"
                    )
                with Image.open(mask_path) as m_img:
                    m_img.verify()
                with Image.open(mask_path) as m_img:
                    m_w, m_h = m_img.size
                    mask_arr = np.asarray(m_img.convert("L"), dtype=np.uint8)
                if (m_w, m_h) != (width, height):
                    raise ValueError(
                        f"Mask dimensions {(m_w, m_h)} do not match image {(width, height)} for {img_path}"
                    )
                if not np.any(mask_arr > 0):
                    raise ValueError(f"Ground-truth mask {mask_path} contains no positive anomaly pixels")
                mask_relpath = mask_path.relative_to(dataset_root).as_posix()
                mask_sha256 = compute_file_sha256(mask_path)
                num_test_anomalous += 1
            else:
                num_test_good += 1

            samples.append(
                SampleRecord(
                    sample_id=f"{category}/test/{defect_name}/{img_path.stem}",
                    category=category,
                    split="test",
                    defect_type=defect_name,
                    is_anomalous=is_anom,
                    image_relpath=img_path.relative_to(dataset_root).as_posix(),
                    mask_relpath=mask_relpath,
                    image_sha256=compute_file_sha256(img_path),
                    mask_sha256=mask_sha256,
                    width=width,
                    height=height,
                )
            )

    digest_payload = json.dumps(
        [s.model_dump(mode="json") for s in samples],
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_sha256 = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()

    num_train_fit = sum(1 for s in samples if s.split == "train_fit")
    num_calibration = sum(1 for s in samples if s.split == "calibration")

    manifest = CategoryDatasetManifest(
        category=category,
        dataset_id=dataset_cfg.hf_dataset_id,
        dataset_revision=dataset_cfg.hf_revision,
        license_id=dataset_cfg.license_id,
        created_at=datetime.now(UTC),
        seed=dataset_cfg.seed,
        calibration_fraction=dataset_cfg.calibration_fraction,
        num_train_fit=num_train_fit,
        num_calibration=num_calibration,
        num_test_good=num_test_good,
        num_test_anomalous=num_test_anomalous,
        defect_types=defect_types,
        manifest_sha256=manifest_sha256,
        samples=samples,
    )

    verify_no_split_leakage(manifest)

    if manifest_out_path is not None:
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_out_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return manifest


def verify_no_split_leakage(manifest: CategoryDatasetManifest) -> None:
    """Strictly verify zero sample path or SHA-256 overlap across train_fit, calibration, and test splits."""
    by_split: dict[str, set[str]] = {"train_fit": set(), "calibration": set(), "test": set()}
    sha_by_split: dict[str, set[str]] = {"train_fit": set(), "calibration": set(), "test": set()}

    for sample in manifest.samples:
        if sample.split in ("train_fit", "calibration") and sample.is_anomalous:
            raise ValueError(f"Anomalous sample {sample.sample_id} leaked into {sample.split} split!")
        by_split[sample.split].add(sample.image_relpath)
        sha_by_split[sample.split].add(sample.image_sha256)

    for split_a, split_b in [("train_fit", "calibration"), ("train_fit", "test"), ("calibration", "test")]:
        path_overlap = by_split[split_a] & by_split[split_b]
        if path_overlap:
            raise ValueError(f"Path leakage detected between {split_a} and {split_b}: {path_overlap}")
        sha_overlap = sha_by_split[split_a] & sha_by_split[split_b]
        if sha_overlap:
            raise ValueError(f"Content SHA-256 leakage detected between {split_a} and {split_b}: {sha_overlap}")


class MVTecCategoryDataset:
    """In-memory index over a validated CategoryDatasetManifest providing split-filtered access."""

    def __init__(self, dataset_root: Path, manifest: CategoryDatasetManifest) -> None:
        self.dataset_root = dataset_root
        self.manifest = manifest

    def get_split(self, split: str) -> list[SampleRecord]:
        return [s for s in self.manifest.samples if s.split == split]

    def load_sample_image(self, record: SampleRecord) -> np.ndarray:
        return load_image_rgb(self.dataset_root / record.image_relpath)

    def load_sample_mask(self, record: SampleRecord) -> np.ndarray:
        if record.mask_relpath is None:
            return np.zeros((record.height, record.width), dtype=np.uint8)
        return load_mask_binary(
            self.dataset_root / record.mask_relpath,
            target_shape=(record.height, record.width),
        )
