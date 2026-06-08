"""
run_phase3.py
=============
Phase 3 master runner — trains the model and optionally starts the server.

    ipl_venv/bin/python run_phase3.py                    # train only
    ipl_venv/bin/python run_phase3.py --serve            # train + API on port 8000
    ipl_venv/bin/python run_phase3.py --serve --port=8080 # custom port
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from loguru import logger


def get_port() -> int:
    for arg in sys.argv:
        if arg.startswith("--port="):
            return int(arg.split("=")[1])
    return 8000


def main():
    logger.info("=== Phase 3: Model Training + Serving ===")

    from src.models.train import run as train
    train()

    if "--serve" in sys.argv:
        import uvicorn
        port = get_port()
        logger.info(f"\nStarting API server at http://localhost:{port}")
        logger.info(f"Docs: http://localhost:{port}/docs")
        uvicorn.run(
            "src.models.serve:app",
            host="0.0.0.0",
            port=port,
            reload=False,
        )


if __name__ == "__main__":
    main()