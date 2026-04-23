"""向后兼容：统一检测入口定义于 ``sql_injection_detector``。"""
from __future__ import annotations

from detection.sql_injection_detector import (
    MergeMode,
    detect_vulnerability,
    detect_vulnerability_json,
)

__all__ = ["MergeMode", "detect_vulnerability", "detect_vulnerability_json"]
