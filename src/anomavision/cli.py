"""CLI orchestration entry point for AnomaVision pipelines, API serving, and Gradio UI."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from anomavision.config import PipelineConfig
from anomavision.pipeline.tracking import (
    log_experiment_to_mlflow,
    record_modal_cost_estimate,
    write_dvc_and_lineage_manifests,
)
from anomavision.schemas import AblationPoint, CategoryEvaluationReport

app = typer.Typer(
    name="anomavision",
    help="AnomaVision: Production-Grade Visual Anomaly Intelligence CLI.",
    add_completion=False,
)
console = Console()


@app.command("run-pipeline")
def run_pipeline_cmd(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML pipeline configuration.",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        help="Optional single MVTec AD category to execute (defaults to all configured categories).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force recomputation of cached embeddings and memory banks.",
    ),
    run_ablations: bool = typer.Option(
        True,
        "--ablations/--no-ablations",
        help="Run mathematical ablation experiments on cached embeddings.",
    ),
) -> None:
    """Execute the end-to-end GPU pipeline on Modal, sync artifacts, and log to MLflow + DVC."""
    from anomavision.modal_app import app as modal_app
    from anomavision.modal_app import execute_remote_gpu_pipeline

    cfg = PipelineConfig.from_yaml(config) if config.is_file() else PipelineConfig()
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_categories = [category] if category else cfg.dataset.categories

    console.print(
        f"[bold cyan]Starting AnomaVision Modal GPU Pipeline[/bold cyan] "
        f"(fingerprint=[yellow]{cfg.fingerprint()}[/yellow], categories={target_categories})"
    )

    config_payload = json.loads(cfg.model_dump_json())
    with modal_app.run():
        result = execute_remote_gpu_pipeline.remote(
            config_dict=config_payload,
            force=force,
            run_ablations=run_ablations,
            categories=target_categories,
        )

    # Unpack bundles and manifests into local artifacts directory
    tar_bytes: bytes = result["bundles_tar_gz"]
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        tar.extractall(path=artifacts_dir, filter="data")

    reports = [CategoryEvaluationReport.model_validate(r) for r in result["reports"]]
    ablations = [AblationPoint.model_validate(a) for a in result["ablations"]]

    summary_payload = {
        "config_fingerprint": cfg.fingerprint(),
        "provenance": result["provenance"],
        "reports": [r.model_dump(mode="json") for r in reports],
        "ablations": [a.model_dump(mode="json") for a in ablations],
    }
    (artifacts_dir / "evaluation_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    started_dt = datetime.fromisoformat(result["started_at"])
    completed_dt = datetime.fromisoformat(result["completed_at"])
    cost_rec = record_modal_cost_estimate(
        job_name=f"modal_gpu_pipeline_{cfg.fingerprint()[:8]}",
        started_at=started_dt,
        completed_at=completed_dt,
        hardware_type=result["hardware_type"],
        hourly_rate_usd=cfg.modal.gpu_hourly_rate_usd,
        artifacts_root=artifacts_dir,
    )

    mlflow_run_id = log_experiment_to_mlflow(
        cfg=cfg,
        reports=reports,
        ablations=ablations,
        artifacts_root=artifacts_dir,
    )
    lineage_path = write_dvc_and_lineage_manifests(
        cfg=cfg,
        reports=reports,
        artifacts_root=artifacts_dir,
        project_root=Path.cwd(),
    )

    table = Table(title="AnomaVision MVTec AD Quantitative Results")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Image AUROC (95% CI)", style="green")
    table.add_column("Pixel AUROC (95% CI)", style="green")
    table.add_column("AUPRO", style="magenta")
    table.add_column("F1 (Unsup / Oracle)", style="yellow")
    table.add_column("Baseline AUROC", style="dim")
    table.add_column("Latency (ms)", style="blue")
    table.add_column("Coreset (MB)", style="white")

    for rep in reports:
        table.add_row(
            rep.category,
            f"{rep.image_auroc.value:.4f} [{rep.image_auroc.ci_lower:.4f}, {rep.image_auroc.ci_upper:.4f}]",
            f"{rep.pixel_auroc.value:.4f} [{rep.pixel_auroc.ci_lower:.4f}, {rep.pixel_auroc.ci_upper:.4f}]",
            f"{rep.aupro:.4f}",
            f"{rep.f1_at_threshold:.4f} / {rep.f1_oracle_upper_bound:.4f}",
            f"{rep.baseline_image_auroc:.4f}",
            f"{rep.inference_latency_ms_mean:.1f}",
            f"{rep.memory_bank_size_mb:.2f}",
        )

    console.print(table)
    console.print(
        f"[bold green]Pipeline Complete![/bold green] "
        f"Duration: {cost_rec.duration_seconds:.1f}s | "
        f"Estimated Modal L4 Cost: [bold yellow]${cost_rec.estimated_cost_usd:.4f} USD[/bold yellow] | "
        f"MLflow Run: {mlflow_run_id} | Lineage: {lineage_path}"
    )


@app.command("serve-api")
def serve_api_cmd(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    bundles_dir: Path = typer.Option(Path("artifacts/bundles"), "--bundles-dir"),
) -> None:
    """Launch the production FastAPI inference service."""
    import os

    import uvicorn

    os.environ["ANOMAVISION_BUNDLES_DIR"] = str(bundles_dir.resolve())
    uvicorn.run("anomavision.api.app:create_app", host=host, port=port, factory=True)


@app.command("launch-ui")
def launch_ui_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    bundles_dir: Path = typer.Option(Path("artifacts/bundles"), "--bundles-dir"),
    share: bool = typer.Option(False, "--share"),
) -> None:
    """Launch the interactive Gradio industrial inspection application."""
    from anomavision.ui.gradio_app import build_gradio_interface

    demo = build_gradio_interface(bundles_root=bundles_dir)
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    app()
