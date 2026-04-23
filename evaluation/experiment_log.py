"""评测与实验运行日志：写入 logs/ 下，避免污染仓库根说明。"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_file_logging(
    log_dir: Path,
    *,
    name: str = "eval",
    level: int = logging.INFO,
) -> Path:
    """
    配置根 logger 的一个 FileHandler，返回日志文件路径。

    典型目录：logs/experiments、logs/errors
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{name}_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    root = logging.getLogger()
    root.setLevel(min(root.level or logging.WARNING, level))
    root.addHandler(fh)

    logging.getLogger("evaluation").info("log file: %s", path)
    return path
