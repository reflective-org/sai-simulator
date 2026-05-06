variable "aws_region" {
  type        = string
  description = "AWS region for the runner instance and IAM resources."
  default     = "us-west-2"
}

variable "project_name" {
  type        = string
  description = "Prefix for resource names (tags, IAM, SG)."
  default     = "github-sai-validation-runner"
}

variable "vpc_id" {
  type        = string
  default     = null
  nullable    = true
  description = "VPC where the runner will live. Leave null to use the account default VPC in var.aws_region."
}

variable "subnet_id" {
  type        = string
  default     = null
  nullable    = true
  description = "Subnet for the runner. Leave null to pick the first subnet (stable sort by ID) in the chosen VPC with map-public-ip-on-launch=true. Set explicitly for private subnets or when the filter matches nothing."
}

variable "associate_public_ip" {
  type        = bool
  description = "Assign a public IPv4 on launch. Default true when using auto-selected default-VPC public subnets; set false for private subnets with NAT."
  default     = true
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type (validation is memory-heavy; avoid nano/micro)."
  default     = "t3.xlarge"
}

variable "root_volume_size_gb" {
  type        = number
  description = "Root gp3 volume size in GB."
  default     = 200
}

variable "enable_ec2_instance_connect" {
  type        = bool
  description = "Allow inbound SSH (port 22) from the regional EC2 Instance Connect managed prefix list so you can use Connect in the console or ec2-instance-connect CLI. Needs a reachable path to the instance (e.g. public IP); private-only setups need an Instance Connect Endpoint (not created here)."
  default     = true
}

variable "key_name" {
  type        = string
  description = "Optional EC2 key pair name for SSH. Leave empty to use SSM Session Manager or EC2 Instance Connect."
  default     = ""
}

variable "management_cidr_blocks" {
  type        = list(string)
  description = "CIDRs allowed to SSH to the instance (port 22). Ignored if key_name is empty."
  default     = []
}

variable "validation_s3_bucket" {
  type        = string
  description = "S3 bucket name holding validation reference GeoTIFFs (used for IAM read policy)."
}

variable "validation_s3_prefix" {
  type        = string
  description = "Optional key prefix inside the bucket (no leading slash). Empty means entire bucket for List/Get patterns."
  default     = ""
}

variable "actions_runner_version" {
  type        = string
  description = "GitHub Actions runner release version (see https://github.com/actions/runner/releases)."
  default     = "2.329.0"
}

variable "github_repo_owner" {
  type        = string
  description = "GitHub org or user that owns the repository (for output instructions only)."
}

variable "github_repo_name" {
  type        = string
  description = "Repository name (for output instructions only)."
}

variable "runner_labels" {
  type        = string
  description = "Comma-separated labels to pass to config.sh (must include workflow runs-on labels)."
  default     = "sai-simulator-validation"
}

variable "extra_tags" {
  type        = map(string)
  description = "Extra tags applied to the instance and supporting resources."
  default     = {}
}
