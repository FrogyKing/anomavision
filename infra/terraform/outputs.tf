output "ecr_repository_url" {
  description = "ECR repository URL for AnomaVision container images."
  value       = aws_ecr_repository.anomavision.repository_url
}

output "artifacts_bucket_name" {
  description = "S3 bucket name for packaged MVTec AD memory bank bundles."
  value       = aws_s3_bucket.artifacts.id
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "fargate_cpu_task_arn" {
  description = "ARN of the CPU Fargate ECS task definition."
  value       = aws_ecs_task_definition.fargate_cpu.arn
}
