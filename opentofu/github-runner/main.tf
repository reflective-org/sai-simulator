data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_s3_bucket" "validation" {
  bucket = var.validation_s3_bucket
}

data "aws_ec2_managed_prefix_list" "ec2_instance_connect" {
  count = var.enable_ec2_instance_connect ? 1 : 0
  name  = "com.amazonaws.${var.aws_region}.ec2-instance-connect"
}

data "aws_vpc" "default" {
  count   = var.vpc_id == null ? 1 : 0
  default = true
}

locals {
  vpc_id_resolved = coalesce(var.vpc_id, try(data.aws_vpc.default[0].id, null))
}

data "aws_subnets" "public_in_vpc" {
  filter {
    name   = "vpc-id"
    values = [local.vpc_id_resolved]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

locals {
  public_subnet_ids_sorted = sort(data.aws_subnets.public_in_vpc.ids)
  subnet_id_resolved = coalesce(
    var.subnet_id,
    length(local.public_subnet_ids_sorted) > 0 ? local.public_subnet_ids_sorted[0] : null,
  )
  github_repo_full_name = "${var.github_repo_owner}/${var.github_repo_name}"

  base_tags = merge(
    {
      Project   = var.project_name
      ManagedBy = "opentofu"
      Purpose   = "github-actions-self-hosted-runner"
    },
    var.extra_tags,
  )

  prefix_trim = trim(var.validation_s3_prefix, "/")
  object_arn  = local.prefix_trim == "" ? "${data.aws_s3_bucket.validation.arn}/*" : "${data.aws_s3_bucket.validation.arn}/${local.prefix_trim}/*"

  validation_s3_policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid      = "GetValidationObjects"
          Effect   = "Allow"
          Action   = ["s3:GetObject"]
          Resource = [local.object_arn]
        }
      ],
      [
        merge(
          {
            Sid      = local.prefix_trim == "" ? "ListValidationBucket" : "ListValidationPrefix"
            Effect   = "Allow"
            Action   = ["s3:ListBucket"]
            Resource = [data.aws_s3_bucket.validation.arn]
          },
          local.prefix_trim == "" ? {} : {
            Condition = {
              StringLike = {
                "s3:prefix" = [
                  local.prefix_trim,
                  "${local.prefix_trim}/*",
                ]
              }
            }
          }
        )
      ]
    )
  })

  user_data = <<-EOT
    #!/bin/bash
    set -euo pipefail
    exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

    dnf update -y
    dnf install -y git tar gzip libicu ec2-instance-connect python3 python3-pip gfortran

    RUNNER_USER="ec2-user"

    RUN_DIR="/opt/actions-runner"
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"

    curl -fsSL -o actions-runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${var.actions_runner_version}/actions-runner-linux-x64-${var.actions_runner_version}.tar.gz"
    tar xzf actions-runner.tar.gz
    rm -f actions-runner.tar.gz
    chown -R "$RUNNER_USER:$RUNNER_USER" "$RUN_DIR"

    sudo -u "$RUNNER_USER" bash -c 'cd /opt/actions-runner && ./bin/installdependencies.sh' || true

    cat >/etc/motd.d/github-runner.txt <<MOTD
    GitHub Actions runner is in /opt/actions-runner

    Register with a new runner token (GitHub → Settings → Actions → Runners → New self-hosted runner):

      sudo -u ec2-user bash
      cd /opt/actions-runner
      ./config.sh --url https://github.com/${var.github_repo_owner}/${var.github_repo_name} --token TOKEN --labels ${var.runner_labels}
      sudo ./svc.sh install ec2-user
      sudo ./svc.sh start

    Match --labels to .github/workflows/s3-validation-self-hosted.yml runs-on (self-hosted is added by GitHub; include sai-simulator-validation).
    MOTD

    echo "User-data finished. Register the runner using a token from GitHub."
    EOT
}

resource "aws_security_group" "runner" {
  name        = "${var.project_name}-sg"
  description = "GitHub Actions self-hosted runner"
  vpc_id      = local.vpc_id_resolved

  egress {
    description      = "All outbound (GitHub, S3, package repos)"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  dynamic "ingress" {
    for_each = var.enable_ec2_instance_connect ? [1] : []
    content {
      description     = "SSH from EC2 Instance Connect (console / CLI)"
      from_port       = 22
      to_port         = 22
      protocol        = "tcp"
      prefix_list_ids = [data.aws_ec2_managed_prefix_list.ec2_instance_connect[0].id]
    }
  }

  dynamic "ingress" {
    for_each = var.key_name != "" && length(var.management_cidr_blocks) > 0 ? [1] : []
    content {
      description = "SSH from management CIDRs"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.management_cidr_blocks
    }
  }

  tags = local.base_tags
}

resource "aws_iam_role" "runner" {
  name = "${var.project_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = local.base_tags
}

data "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_role" "github_actions" {
  name = "${var.project_name}-github-actions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = data.aws_iam_openid_connect_provider.github_actions.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          # Only the subjects the validation workflows run under: PRs, plus main/public
          # for the watchdog (schedule/workflow_run) and manual dispatch. Workflows on any
          # other branch or tag cannot assume this role.
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
            "token.actions.githubusercontent.com:sub" = [
              "repo:${local.github_repo_full_name}:pull_request",
              "repo:${local.github_repo_full_name}:ref:refs/heads/main",
              "repo:${local.github_repo_full_name}:ref:refs/heads/public",
            ]
          }
        }
      }
    ]
  })

  tags = local.base_tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "validation_s3_read" {
  name   = "${var.project_name}-validation-s3-read"
  role   = aws_iam_role.runner.id
  policy = local.validation_s3_policy
}

resource "aws_iam_instance_profile" "runner" {
  name = "${var.project_name}-profile"
  role = aws_iam_role.runner.name

  tags = local.base_tags
}

resource "aws_iam_role_policy" "github_actions_ec2_control" {
  name = "${var.project_name}-github-actions-ec2-control"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DescribeEc2ForValidationRunner"
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceStatus",
        ]
        Resource = "*"
      },
      {
        Sid    = "StartStopValidationRunner"
        Effect = "Allow"
        Action = [
          "ec2:StartInstances",
          "ec2:StopInstances",
        ]
        Resource = [aws_instance.runner.arn]
      },
      {
        Sid    = "TagValidationRunner"
        Effect = "Allow"
        Action = [
          "ec2:CreateTags",
        ]
        Resource = [aws_instance.runner.arn]
        Condition = {
          "ForAllValues:StringEquals" = {
            "aws:TagKeys" = [
              "gh_validation_lease_expires_at",
              "gh_validation_repo",
              "gh_validation_pr",
            ]
          }
        }
      },
    ]
  })
}

resource "aws_instance" "runner" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = local.subnet_id_resolved
  vpc_security_group_ids = [aws_security_group.runner.id]
  iam_instance_profile   = aws_iam_instance_profile.runner.name

  associate_public_ip_address = var.associate_public_ip

  key_name = var.key_name != "" ? var.key_name : null

  user_data                   = base64encode(local.user_data)
  user_data_replace_on_change = true

  lifecycle {
    # Keep the long-lived runner from being rebuilt by routine drift: the AMI
    # data source tracks the newest AL2023 image, user_data edits only matter for
    # a fresh instance, and the validation workflows stamp gh_validation_* lease
    # tags at runtime. Replace deliberately with `tofu apply -replace=aws_instance.runner`.
    ignore_changes = [
      ami,
      user_data,
      tags["gh_validation_hold_until"],
      tags["gh_validation_lease_expires_at"],
      tags["gh_validation_pr"],
      tags["gh_validation_repo"],
    ]

    precondition {
      condition     = local.subnet_id_resolved != null
      error_message = "No subnet selected. Set subnet_id, or use a VPC that has subnets with map-public-ip-on-launch=true (typical for default VPC public subnets)."
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gb
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tags = merge(local.base_tags, { Name = "${var.project_name}-ec2" })
}
