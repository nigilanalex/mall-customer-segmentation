"""Prepare and start the project with one command: python run.py."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ENV = BASE / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_step(arguments):
    """Use argument lists so project paths containing spaces work safely."""
    subprocess.run([str(item) for item in arguments], cwd=BASE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port (default: 8501)")
    parser.add_argument("--retrain", action="store_true", help="Refresh models and outputs before starting")
    parser.add_argument("--prepare-only", action="store_true", help="Set up the project without starting the server")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    try:
        new_environment = not PYTHON.exists()
        if new_environment:
            print("Creating the project virtual environment...", flush=True)
            run_step([sys.executable, "-m", "venv", ENV])
        print("Checking and installing required packages...", flush=True)
        run_step([PYTHON, "-m", "pip", "install", "--disable-pip-version-check", "-r", BASE / "requirements.txt"])

        # Fresh installations rebuild bundled models in their own Python environment.
        required = [BASE / "models/analytics_bundle.pkl", BASE / "database/customers.db",
                    BASE / "dataset/Mall_Customers.csv", BASE / "dataset/Transactions.csv"]
        if args.retrain or new_environment or any(not path.exists() for path in required):
            print("Preparing the analytics database, models and graphs...", flush=True)
            run_step([PYTHON, BASE / "main.py"])
        if args.prepare_only:
            print("Project ready. Run python run.py to open the dashboard.")
            return 0
        print(f"Dashboard: http://localhost:{args.port} | Demo login: admin / admin123", flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
        run_step([PYTHON, "-m", "streamlit", "run", BASE / "app.py",
                  "--server.address", "127.0.0.1", "--server.port", args.port])
        return 0
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Startup failed: {exc}\nFix the error shown above, then run the same command again.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
