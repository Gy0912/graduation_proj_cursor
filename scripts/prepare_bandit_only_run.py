from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src_path = ROOT / "configs" / "default_run.yaml"
    dst_path = ROOT / "configs" / "default_bandit_only_run.yaml"

    if not src_path.exists():
        raise FileNotFoundError(f"missing: {src_path} (run prepare_default_run.py first)")

    cfg = yaml.safe_load(src_path.read_text(encoding="utf-8"))

    outputs = cfg.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("configs/default_run.yaml: outputs must be a dict")

    renamed: dict[str, object] = {}
    for k, v in outputs.items():
        if isinstance(v, str) and v.endswith(".json"):
            renamed[k] = v[: -len(".json")] + "_bandit_only.json"
        else:
            renamed[k] = v

    cfg["outputs"] = renamed
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"[OK] wrote {dst_path}")


if __name__ == "__main__":
    main()

