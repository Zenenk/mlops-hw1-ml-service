from __future__ import annotations

from pathlib import Path
import re


PB2_GRPC = Path("ml_service/grpc/ml_service_pb2_grpc.py")


def main() -> None:
    if not PB2_GRPC.exists():
        raise SystemExit(f"File not found: {PB2_GRPC}")

    t = PB2_GRPC.read_text()

    # 1) Если патч применяли дважды: "from . from . import ..." -> "from . import ..."
    t = re.sub(
        r"(?m)^\s*from\s+\.\s+from\s+\.\s+import\s+ml_service_pb2\s+as\s+ml__service__pb2\s*(#.*)?$",
        r"from . import ml_service_pb2 as ml__service__pb2\1",
        t,
    )

    # 2) Абсолютный импорт (возможен хвост-комментарий)
    t = re.sub(
        r"(?m)^\s*import\s+ml_service_pb2\s+as\s+ml__service__pb2\s*(#.*)?$",
        r"from . import ml_service_pb2 as ml__service__pb2\1",
        t,
    )

    # 3) Иногда генератор может написать так (редко, но безопасно обработать)
    t = re.sub(
        r"(?m)^\s*import\s+ml_service\.grpc\.ml_service_pb2\s+as\s+ml__service__pb2\s*(#.*)?$",
        r"from . import ml_service_pb2 as ml__service__pb2\1",
        t,
    )

    # 4) Убрать warnings (чтобы ruff не ругался), если он там есть
    t = t.replace("import warnings\n", "")

    PB2_GRPC.write_text(t)

    new = PB2_GRPC.read_text()
    if re.search(r"(?m)^\s*import\s+ml_service_pb2\s+as\s+ml__service__pb2", new):
        raise SystemExit("patch failed: absolute import still present")
    if re.search(r"(?m)^\s*from\s+\.\s+from\s+\.\s+import", new):
        raise SystemExit("patch failed: duplicated 'from . from .' still present")

    print(f"patched: {PB2_GRPC}")


if __name__ == "__main__":
    main()
