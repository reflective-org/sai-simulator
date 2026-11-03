terraform {
  required_version = ">= 1.6.0"

  # Partial backend config: region, bucket and key come from backend.hcl
  # (see backend.hcl.example). Pass it at init: tofu init -backend-config=backend.hcl
  backend "s3" {
    encrypt = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
