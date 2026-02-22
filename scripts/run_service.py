# scripts/run_service.py
from __future__ import annotations

import subprocess
import sys

import uvicorn

from ml_service.config import settings


def run_rest() -> None:
    uvicorn.run(
        "ml_service.api_rest:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def run_grpc() -> None:
    from ml_service.grpc.server import serve
    serve(port=50051)


def run_dashboard() -> None:
    app_path = settings.root_dir / "dashboard" / "app.py"
    cmd = ["streamlit", "run", str(app_path)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python -m scripts.run_service [rest|grpc|dashboard]")
        raise SystemExit(1)

    mode = args[0]
    if mode == "rest":
        run_rest()
    elif mode == "grpc":
        run_grpc()
    elif mode == "dashboard":
        run_dashboard()
    else:
        print(f"Unknown mode: {mode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()