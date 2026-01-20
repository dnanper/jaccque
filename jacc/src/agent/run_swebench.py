#!/usr/bin/env python3
"""Run Jacc agent on SWE-bench instances.

Usage:
    python -m agent.run_swebench --subset lite --split dev --slice "0:1" -m gemini/gemini-2.5-flash -o ./output

With Memory Integration:
    # Step 1: Start memory database (one-time or if not running)
    cd jacc/src/memory-api
    docker-compose -f docker-compose.dev.yml up -d
    
    # Step 2: Run with --memory flag
    python -m agent.run_swebench --memory --subset lite --split dev --slice "0:1" -o ./output
"""

import argparse
import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path

import yaml
from datasets import load_dataset

from agent import AgentConfig, run_agent
from agent.models import get_model
from agent.environments import get_environment
from agent.memory import get_memory_client, MemoryConfig, NoOpMemoryClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SWE-bench dataset mappings
DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def get_swebench_docker_image(instance: dict) -> str:
    """Get Docker image name for a SWE-bench instance."""
    # Same logic as mini-swe-agent
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        # Docker doesn't allow double underscore
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def filter_instances(
    instances: list[dict],
    filter_spec: str = "",
    slice_spec: str = "",
) -> list[dict]:
    """Filter and slice instances."""
    # Filter by regex
    if filter_spec:
        before = len(instances)
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
        logger.info(f"Filter: {before} -> {len(instances)} instances")
    
    # Slice
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        logger.info(f"Slice '{slice_spec}': {len(instances)} instances")
    
    return instances


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update predictions JSON file."""
    preds_file = output_path / "preds.json"
    output_data = {}
    if preds_file.exists():
        output_data = json.loads(preds_file.read_text())
    
    output_data[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }
    preds_file.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    config: AgentConfig,
    model_name: str,
    provider: str,
    timeout: int,
    output_dir: Path,
    memory_client = None,
) -> tuple[str, str]:
    """Process a single SWE-bench instance."""
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]
    
    logger.info(f"Processing: {instance_id}")
    logger.info(f"Problem: {problem_statement[:200]}...")
    
    # Create model
    model = get_model(provider, model_name=model_name)
    
    # Create Docker environment
    docker_image = get_swebench_docker_image(instance)
    logger.info(f"Docker image: {docker_image}")
    
    env = get_environment({
        "type": "docker",
        "image": docker_image,
        "timeout": timeout,
        "cwd": "/testbed",
    })
    
    exit_status, result = None, None
    try:
        final_state = run_agent(
            problem_statement, 
            config, 
            model, 
            env,
            memory_client=memory_client,
            instance_id=instance_id,
        )
        exit_status = final_state.get("exit_status", "unknown")
        
        # Get the patch (git diff)
        diff_result = env.execute("git diff")
        result = diff_result.get("output", "")
        
        # Save trajectory
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        traj_file = instance_dir / f"{instance_id}.traj.json"
        trajectory = {
            "instance_id": instance_id,
            "exit_status": exit_status,
            "steps": final_state.get("step_count"),
            "cost": final_state.get("total_cost"),
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in final_state.get("messages", [])
            ],
        }
        traj_file.write_text(json.dumps(trajectory, indent=2))
        
        logger.info(f"Finished {instance_id}: {exit_status}, steps={final_state.get('step_count')}")
        
    except Exception as e:
        exit_status = type(e).__name__
        result = ""
        logger.exception(f"Error processing {instance_id}: {e}")
    finally:
        env.cleanup()
    
    return exit_status, result


def main():
    parser = argparse.ArgumentParser(description="Run Jacc agent on SWE-bench")
    
    # Data selection
    parser.add_argument("--subset", type=str, default="lite", help="SWE-bench subset")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split")
    parser.add_argument("--slice", type=str, default="", help="Slice spec (e.g., '0:1')")
    parser.add_argument("--filter", type=str, default="", help="Filter by regex")
    
    # Model
    parser.add_argument("-m", "--model", type=str, default="gemini/gemini-2.5-flash", help="Model name")
    parser.add_argument("--provider", type=str, default="api", help="Model provider")
    
    # Config
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    parser.add_argument("--timeout", type=int, default=120, help="Command timeout")
    
    # Output
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    
    # Memory
    parser.add_argument("--memory", action="store_true", help="Enable memory integration")
    parser.add_argument("--memory-bank", type=str, default="swe_agent", help="Memory bank ID")
    
    # Other
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--redo-existing", action="store_true", help="Redo existing instances")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    if args.config:
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig.default()
    
    # Initialize memory client if enabled
    memory_client = None
    if args.memory:
        from agent.memory import run_async
        logger.info("Initializing memory client...")
        logger.info("Make sure memory database is running:")
        logger.info("  cd jacc/src/memory-api && docker-compose -f docker-compose.dev.yml up -d")
        
        try:
            memory_config = MemoryConfig(bank_id=args.memory_bank)
            memory_client = get_memory_client(memory_config, mode="direct")
            run_async(memory_client.initialize(), timeout=60)
            logger.info(f"Memory enabled: bank_id={args.memory_bank}")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            logger.error("")
            logger.error("=" * 60)
            logger.error("MEMORY DATABASE NOT AVAILABLE")
            logger.error("=" * 60)
            logger.error("To start the memory database:")
            logger.error("  cd jacc/src/memory-api")
            logger.error("  docker-compose -f docker-compose.dev.yml up -d")
            logger.error("")
            logger.error("Then retry with --memory flag.")
            logger.error("=" * 60)
            logger.warning("Continuing without memory integration...")
            memory_client = NoOpMemoryClient()
    
    # Load dataset
    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    logger.info(f"Loading dataset: {dataset_path}, split: {args.split}")
    instances = list(load_dataset(dataset_path, split=args.split))
    
    # Filter and slice
    instances = filter_instances(instances, args.filter, args.slice)
    
    # Skip existing
    preds_file = output_dir / "preds.json"
    if not args.redo_existing and preds_file.exists():
        existing = set(json.loads(preds_file.read_text()).keys())
        instances = [i for i in instances if i["instance_id"] not in existing]
        logger.info(f"Skipped {len(existing)} existing instances")
    
    logger.info(f"Running on {len(instances)} instances")
    
    # Process each instance
    try:
        for instance in instances:
            instance_id = instance["instance_id"]
            
            try:
                exit_status, result = process_instance(
                    instance=instance,
                    config=config,
                    model_name=args.model,
                    provider=args.provider,
                    timeout=args.timeout,
                    output_dir=output_dir,
                    memory_client=memory_client,
                )
                
                update_preds_file(output_dir, instance_id, args.model, result)
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.exception(f"Uncaught error for {instance_id}: {e}")
                continue
    finally:
        # Cleanup memory client
        if memory_client and hasattr(memory_client, 'close'):
            try:
                from agent.memory import run_async, close_async_runner
                run_async(memory_client.close(), timeout=5)
                close_async_runner()
                logger.info("Memory client closed")
            except Exception:
                pass
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

