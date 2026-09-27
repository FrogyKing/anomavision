# AWS Production Deployment Blueprint for AnomaVision

While **Modal Serverless GPU (NVIDIA L4)** serves as the primary active execution environment for AnomaVision's representation extraction, coreset construction, and benchmarking, the system is designed for straightforward deployment to AWS using container-native primitives.

## Architectural Separation: CPU vs. GPU Serving on AWS

A critical infrastructure constraint in AWS is that **AWS Fargate does not support GPU-accelerated tasks**. Therefore, AnomaVision separates its AWS deployment topology into two distinct operational tiers depending on plant inspection throughput requirements:

```
                         +-----------------------------------+
                         |      Application Load Balancer    |
                         +-----------------+-----------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v (Option A: Low/Moderate RPS)                v (Option B: High RPS / SAM 3)
     +------------------------------+              +------------------------------+
     |   AWS ECS on Fargate (CPU)   |              |  AWS ECS on EC2 GPU Cluster  |
     |  - 4 vCPU / 8 GB RAM         |              |  - g6.xlarge (NVIDIA L4 GPU) |
     |  - DINOv3 ViT-S/16 (Frozen)  |              |  - DINOv3 + SAM 3 BFloat16   |
     |  - FAISS CPU Coreset Search  |              |  - CUDA Tensor Core k-NN     |
     |  - ~180-250 ms / inspection  |              |  - ~33-55 ms / inspection    |
     +--------------+---------------+              +--------------+---------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |   Amazon S3 Artifact Bucket       |
                         |   (Packaged Coreset + FAISS       |
                         |    + Calibration Bundles)         |
                         +-----------------------------------+
```

### Option A: CPU-Backed Serving (`ECS + Fargate`)
- **When to use**: Edge/line inspection cells with `< 4 RPS` using DINOv3 ViT-S/16 (`21.6M` parameters) + FAISS coreset scoring without SAM 3 (`860M` parameters).
- **Characteristics**:
  - Zero EC2 instance management overhead.
  - Because PatchCore requires **no training at inference time** and our greedy $k$-center coreset compresses the memory bank to `4–21 MB`, the entire memory bank and FAISS index reside in L3 cache / RAM.
  - Configured via `enable_gpu_capacity_provider = false` in `infra/terraform/variables.tf`.

### Option B: GPU-Backed Serving (`ECS on EC2 g6.xlarge` with NVIDIA L4)
- **When to use**: High-speed optical inspection lines (`15–30+ FPS`) or deployments enabling real-time **SAM 3** (`tkclam/sam3`) foreground object gating.
- **Characteristics**:
  - Uses an ECS Capacity Provider backed by an EC2 Auto Scaling Group of `g6.xlarge` instances (1x NVIDIA L4 24 GB GPU, matching our Modal L4 benchmark hardware).
  - Delivers `~33 ms` inference latency (DINOv3 + GPU k-NN) or `~55 ms` with SAM 3 object mask refinement.
  - Configured via `enable_gpu_capacity_provider = true` in `infra/terraform/variables.tf`.

## Terraform Blueprint (`infra/terraform/`)

The Terraform configuration in `infra/terraform/` defines:
1. **`aws_ecr_repository.anomavision`**: Container registry with image scanning on push and lifecycle retention policies.
2. **`aws_s3_bucket.artifacts`**: Versioned, SSE-S3 encrypted artifact store for packaged category bundles (`patchcore_coreset.npz`, `reference_images.faiss`, `calibration.json`).
3. **`aws_ecs_cluster.main`**: ECS cluster with Container Insights enabled for Prometheus/CloudWatch telemetry.
4. **`aws_ecs_task_definition.api`**: Parameterized task definition supporting both Fargate CPU mode and EC2 GPU (`aws_ecs_capacity_provider` on `g6.xlarge`) mode.

> **Cost & Safety Note**: No live AWS resources are provisioned by default. To validate the Terraform configuration locally without AWS credentials or charges:
> ```bash
> terraform -chdir=infra/terraform init -backend=false
> terraform -chdir=infra/terraform validate
> ```
