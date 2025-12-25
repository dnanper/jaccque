"""Entry point: python -m agent

Subcommands:
    python -m agent swebench ...  → run_swebench
    python -m agent run ...       → simple task runner
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m agent swebench [args]  - Run on SWE-bench")
        print("  python -m agent run -t 'task'    - Run a simple task")
        sys.exit(0)
    
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove subcommand from argv
    
    if cmd == "swebench":
        from agent.run_swebench import main as swebench_main
        swebench_main()
    elif cmd == "run":
        from agent.run_task import main as task_main
        task_main()
    else:
        print(f"Unknown command: {cmd}")
        print("Available: swebench, run")
        sys.exit(1)

if __name__ == "__main__":
    main()
