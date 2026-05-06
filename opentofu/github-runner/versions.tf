terraform {
  required_version = ">= 1.6.0"

  # Region must match the bucket's AWS region or S3 returns PermanentRedirect (301).
  # Pass it at init: tofu init -backend-config=backend.hcl
  # (see backend.hcl.example).
  backend "s3" {
    region  = "us-west-2"
    bucket  = "reflective-opentofu-state"
    key     = "sai-simulator-internal/github-runner/terraform.tfstate"
    encrypt = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
