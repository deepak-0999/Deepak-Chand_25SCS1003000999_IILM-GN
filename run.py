"""
Run the Lore API server.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --reload       # dev mode with hot reload
"""

import argparse
import uvicorn
from config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Lore API server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev)")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )
