output "vpc_id" {
  description = "VPC used for the runner (default VPC when var.vpc_id is null)."
  value       = local.vpc_id_resolved
}

output "subnet_id" {
  description = "Subnet used for the runner (first map-public subnet when var.subnet_id is null)."
  value       = local.subnet_id_resolved
}

output "instance_id" {
  description = "EC2 instance ID for the self-hosted runner."
  value       = aws_instance.runner.id
}

output "private_ip" {
  description = "Private IPv4 address (use SSM Session Manager or private connectivity)."
  value       = aws_instance.runner.private_ip
}

output "public_ip" {
  description = "Public IPv4 when associate_public_ip is true; otherwise null."
  value       = aws_instance.runner.public_ip
}

output "security_group_id" {
  description = "Security group attached to the runner."
  value       = aws_security_group.runner.id
}

output "iam_instance_profile_name" {
  description = "Instance profile name (S3 read + SSM)."
  value       = aws_iam_instance_profile.runner.name
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC (use for VALIDATION_AWS_ROLE_ARN)."
  value       = aws_iam_role.github_actions.arn
}

output "registration_hint" {
  description = "Suggested config.sh URL and labels (token comes from GitHub UI)."
  value = {
    url    = "https://github.com/${var.github_repo_owner}/${var.github_repo_name}"
    labels = var.runner_labels
  }
}
