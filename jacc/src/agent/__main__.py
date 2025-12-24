"""Entry point for running the agent.

Usage:
    python -m agent                           # Interactive mode
    python -m agent -t "Fix the bug"          # With task
    python -m agent --swebench -i instance_id # SWE-bench instance
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from agent import AgentConfig, run_agent
from agent.models import get_model
from agent.environments import get_environment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the SWE agent")
    parser.add_argument("-t", "--task", type=str, help="Task/problem statement")
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    parser.add_argument("-m", "--model", type=str, default="gemini/gemini-2.0-flash", help="Model name")
    parser.add_argument("--provider", type=str, default="api", help="Model provider (api, vllm, hf_inference, local_hf)")
    parser.add_argument("--env", type=str, default="local", help="Environment type (local, docker)")
    parser.add_argument("--docker-image", type=str, help="Docker image (for docker env)")
    parser.add_argument("--timeout", type=int, default=120, help="Command timeout in seconds")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    if args.config:
        config = AgentConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = AgentConfig.default()
        logger.info("Using default config")
    
    # Create model
    model = get_model(args.provider, model_name=args.model)
    logger.info(f"Created {args.provider} model: {args.model}")
    
    # Create environment
    if args.env == "docker":
        if not args.docker_image:
            logger.error("Docker image required for docker environment. Use --docker-image")
            sys.exit(1)
        env = get_environment({
            "type": "docker",
            "image": args.docker_image,
            "timeout": args.timeout,
        })
        logger.info(f"Created docker environment: {args.docker_image}")
    else:
        env = get_environment({
            "type": "local",
            "timeout": args.timeout,
        })
        logger.info("Created local environment")
    
    # Get task
    task = args.task
    if not task:
        print("Enter your task (press Ctrl+D when done):")
        try:
            task = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(0)
    
    if not task:
        logger.error("No task provided")
        sys.exit(1)
    
    logger.info(f"Running agent on task: {task[:100]}...")
    
    # Run agent
    try:
        result = run_agent(task, config, model, env)
        
        print("\n" + "="*60)
        print(f"Exit Status: {result.get('exit_status')}")
        print(f"Exit Message: {result.get('exit_message')}")
        print(f"Steps: {result.get('step_count')}")
        print(f"Total Cost: ${result.get('total_cost', 0):.4f}")
        print("="*60)
        
        # Cleanup environment
        env.cleanup()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        env.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running agent: {e}")
        env.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
