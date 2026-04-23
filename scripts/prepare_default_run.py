from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src_path = ROOT / "configs" / "default.yaml"
    dst_path = ROOT / "configs" / "default_run.yaml"

    if not src_path.exists():
        raise FileNotFoundError(f"missing: {src_path}")

    cfg = yaml.safe_load(src_path.read_text(encoding="utf-8"))

    cfg.setdefault("generation", {})
    cfg["generation"]["temperature"] = 0

    cfg.setdefault("eval", {})
    cfg["eval"]["enable_taint"] = True

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"[OK] wrote {dst_path}")


if __name__ == "__main__":
    main()

