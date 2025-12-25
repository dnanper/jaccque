"""Run agent on a simple task.

Usage:
    python -m agent run -t "List all Python files"
    python -m agent run -t "Fix the bug" --env docker --docker-image python:3.11
"""

import argparse
import logging
import sys

from agent import AgentConfig, run_agent
from agent.models import get_model
from agent.environments import get_environment


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run agent on a task")
    parser.add_argument("-t", "--task", type=str, required=True, help="Task description")
    parser.add_argument("-m", "--model", type=str, default="gemini/gemini-2.0-flash")
    parser.add_argument("--provider", type=str, default="api")
    parser.add_argument("--env", type=str, default="local", choices=["local", "docker"])
    parser.add_argument("--docker-image", type=str, help="Docker image (required if --env docker)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = AgentConfig.default()
    model = get_model(args.provider, model_name=args.model)
    
    if args.env == "docker":
        if not args.docker_image:
            logger.error("--docker-image required for docker environment")
            sys.exit(1)
        env = get_environment({"type": "docker", "image": args.docker_image, "timeout": args.timeout})
    else:
        env = get_environment({"type": "local", "timeout": args.timeout})
    
    try:
        result = run_agent(args.task, config, model, env)
        print(f"\nStatus: {result.get('exit_status')}")
        print(f"Steps: {result.get('step_count')}")
        print(f"Cost: ${result.get('total_cost', 0):.4f}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
