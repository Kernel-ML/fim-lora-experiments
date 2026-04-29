"""Launch FIM-LoRA training jobs on Amazon SageMaker.

Usage:
    # Single job
    uv run python scripts/sagemaker_train.py \
        --method fim_lora \
        --task mnli \
        --rank 8 \
        --seed 42 \
        --instance ml.g5.2xlarge

    # Full GLUE sweep (launches all jobs in parallel)
    uv run python scripts/sagemaker_train.py \
        --sweep glue \
        --instance ml.g5.2xlarge

Instance recommendations:
    ml.g5.2xlarge   — 1x A10G 24GB,  $1.52/hr — GLUE/DeBERTa (recommended)
    ml.g5.12xlarge  — 4x A10G 96GB,  $5.67/hr — faster sweeps
    ml.p3.2xlarge   — 1x V100 16GB,  $3.83/hr — GLUE only (tight on memory)
    ml.p4d.24xlarge — 8x A100 40GB, $32.77/hr — LLaMA-3-8B (overkill for GLUE)
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROLE = os.environ.get("SAGEMAKER_ROLE", "")  # set via env or pass --role
S3_BUCKET = os.environ.get("SAGEMAKER_BUCKET", "")  # your S3 bucket for outputs
ECR_IMAGE = None  # None = use SageMaker HuggingFace DLC (recommended)

GLUE_TASKS = ["mnli", "sst2", "cola", "mrpc", "qqp", "qnli", "stsb", "rte"]
METHODS = ["lora", "adalora", "eva", "fim_lora", "random_rank"]
RANKS = [2, 4, 8, 16]
SEEDS = [42, 1337, 2024]

# SageMaker HuggingFace Deep Learning Container — pick one matching your region
# Full list: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
HF_DLC = {
    "transformers_version": "4.36.0",
    "pytorch_version": "2.1.0",
    "py_version": "py310",
}


# ---------------------------------------------------------------------------
# Launch single job
# ---------------------------------------------------------------------------


def launch_job(
    method: str,
    task: str,
    rank: int,
    seed: int,
    instance_type: str,
    role: str,
    s3_bucket: str,
    wait: bool = False,
) -> str:
    """Launch a single SageMaker training job. Returns the job name."""

    session = sagemaker.Session()
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    job_name = f"fim-lora-{method}-{task}-r{rank}-s{seed}-{timestamp}"

    # Hyperparameters passed as CLI args to train_glue.py
    hyperparameters = {
        "method": method,
        "task": task,
        "rank": str(rank),
        "seed": str(seed),
        "fim-batches": "8",
        "output-dir": "/opt/ml/model",
        "no-wandb": "",  # disable wandb inside SM (use SM metrics instead)
    }

    output_path = f"s3://{s3_bucket}/fim-lora/{method}/{task}/r{rank}/seed{seed}/"

    estimator = HuggingFace(
        entry_point="train_glue.py",
        source_dir="src",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        transformers_version=HF_DLC["transformers_version"],
        pytorch_version=HF_DLC["pytorch_version"],
        py_version=HF_DLC["py_version"],
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name="fim-lora",
        # Log metrics to CloudWatch so SM can plot them
        metric_definitions=[
            {"Name": "eval/accuracy",      "Regex": r"'eval_accuracy':\s*([0-9\.]+)"},
            {"Name": "eval/f1",            "Regex": r"'eval_f1':\s*([0-9\.]+)"},
            {"Name": "eval/matthews",      "Regex": r"'eval_matthews_correlation':\s*([0-9\.]+)"},
            {"Name": "eval/pearson",       "Regex": r"'eval_pearson':\s*([0-9\.]+)"},
            {"Name": "train/loss",         "Regex": r"'loss':\s*([0-9\.]+)"},
        ],
        # Keep instance alive for spot interruption recovery
        use_spot_instances=True,
        max_wait=86400,   # 24 hours
        max_run=43200,    # 12 hours
    )

    estimator.fit(wait=wait, job_name=job_name)
    print(f"  Launched: {job_name}")
    return job_name


# ---------------------------------------------------------------------------
# Sweep launcher
# ---------------------------------------------------------------------------


def launch_glue_sweep(instance_type: str, role: str, s3_bucket: str) -> list[str]:
    """Launch all GLUE jobs in parallel. Returns list of job names."""
    job_names = []
    total = len(METHODS) * len(GLUE_TASKS) * len(RANKS) * len(SEEDS)
    print(f"Launching {total} jobs in parallel...")

    for method in METHODS:
        for task in GLUE_TASKS:
            for rank in RANKS:
                for seed in SEEDS:
                    name = launch_job(
                        method=method, task=task, rank=rank, seed=seed,
                        instance_type=instance_type, role=role,
                        s3_bucket=s3_bucket, wait=False,
                    )
                    job_names.append(name)

    print(f"\nAll {len(job_names)} jobs launched.")
    print("Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs")
    return job_names


def launch_llama_sweep(instance_type: str, role: str, s3_bucket: str) -> list[str]:
    """Launch LLaMA-3-8B commonsense jobs."""
    from sagemaker.huggingface import HuggingFace

    session = sagemaker.Session()
    job_names = []

    for method in ["lora", "adalora", "fim_lora"]:
        for seed in SEEDS:
            timestamp = datetime.now().strftime("%m%d-%H%M%S")
            job_name = f"fim-lora-llama-{method}-r16-s{seed}-{timestamp}"

            estimator = HuggingFace(
                entry_point="train_commonsense.py",
                source_dir="src",
                role=role,
                instance_type=instance_type,
                instance_count=1,
                transformers_version=HF_DLC["transformers_version"],
                pytorch_version=HF_DLC["pytorch_version"],
                py_version=HF_DLC["py_version"],
                hyperparameters={
                    "method": method, "rank": "16", "seed": str(seed),
                    "output-dir": "/opt/ml/model", "no-wandb": "",
                },
                output_path=f"s3://{s3_bucket}/fim-lora/llama/{method}/r16/seed{seed}/",
                base_job_name="fim-lora-llama",
                use_spot_instances=True,
                max_wait=172800,  # 48 hours
                max_run=86400,    # 24 hours
                # LLaMA-3 needs HF token
                environment={"HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", "")},
            )
            estimator.fit(wait=False, job_name=job_name)
            job_names.append(job_name)
            print(f"  Launched: {job_name}")

    return job_names


# ---------------------------------------------------------------------------
# Collect results from S3
# ---------------------------------------------------------------------------


def collect_from_s3(s3_bucket: str, prefix: str = "fim-lora", local_dir: str = "results") -> None:
    """Download all results.json files from S3 to local results/."""
    import subprocess
    cmd = f"aws s3 sync s3://{s3_bucket}/{prefix}/ {local_dir}/ --include '*/results.json'"
    print(f"Syncing results: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"Results downloaded to {local_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--task", choices=GLUE_TASKS)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", choices=["glue", "llama"], help="Launch full sweep")
    parser.add_argument("--collect", action="store_true", help="Download results from S3")
    parser.add_argument("--instance", default="ml.g5.2xlarge",
                        help="SageMaker instance type")
    parser.add_argument("--role", default=ROLE,
                        help="SageMaker IAM role ARN (or set SAGEMAKER_ROLE env var)")
    parser.add_argument("--bucket", default=S3_BUCKET,
                        help="S3 bucket for outputs (or set SAGEMAKER_BUCKET env var)")
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    args = parser.parse_args()

    if not args.role:
        print("ERROR: Set --role or SAGEMAKER_ROLE env var to your SageMaker execution role ARN.")
        exit(1)
    if not args.bucket:
        print("ERROR: Set --bucket or SAGEMAKER_BUCKET env var to your S3 bucket name.")
        exit(1)

    if args.collect:
        collect_from_s3(args.bucket)
    elif args.sweep == "glue":
        launch_glue_sweep(args.instance, args.role, args.bucket)
    elif args.sweep == "llama":
        launch_llama_sweep(args.instance, args.role, args.bucket)
    elif args.method and args.task:
        launch_job(
            method=args.method, task=args.task, rank=args.rank, seed=args.seed,
            instance_type=args.instance, role=args.role,
            s3_bucket=args.bucket, wait=args.wait,
        )
    else:
        parser.print_help()
