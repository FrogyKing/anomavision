variable "aws_region" {
  description = "AWS region for ECR, S3, and ECS resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Resource naming prefix."
  type        = string
  default     = "anomavision"
}

variable "environment" {
  description = "Deployment environment label."
  type        = string
  default     = "prod"
}

variable "image_tag" {
  description = "Immutable Docker image tag in ECR."
  type        = string
  default     = "v1.0.0"
}

variable "fargate_cpu" {
  description = "CPU units for Fargate CPU inference task (4096 = 4 vCPU)."
  type        = number
  default     = 4096
}

variable "fargate_memory" {
  description = "Memory (MiB) for Fargate CPU inference task."
  type        = number
  default     = 8192
}

variable "enable_gpu_capacity_provider" {
  description = "Enable ECS on EC2 GPU (g6.xlarge NVIDIA L4) task definition for GPU-accelerated serving."
  type        = bool
  default     = true
}
