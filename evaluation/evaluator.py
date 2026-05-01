from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from detection.detector import detect_vulnerability
from detection.sql_injection_detector import extract_python_code_with_debug
from evaluation.metrics import (
    MARKER_EXPLANATION,
    MARKER_SAFE_SOLUTION,
    MARKER_WARNING,
    MetricBundle,
    aggregate_metrics,
    print_eval_summary,
)

# 对照实验：恒输出参数化安全片段（不加载大模型）
ALWAYS_SAFE_SYNTHETIC_OUTPUT = '''import pymysql


def fetch_rows(cur, value: str):
    sql = "SELECT * FROM users WHERE username = %s"
    cur.execute(sql, (value,))
    return cur.fetchall()
'''


CRITICAL_SAMPLE_FIELDS: tuple[str, ...] = ("id", "prompt", "expected_vulnerable")
DEBUG_INVALID_SAMPLES_PATH = Path("outputs/debug_invalid_samples.json")
MAX_DEBUG_INVALID_SAMPLES = 20


def _response_structure_flags(raw_output: str | None) -> dict[str, bool]:
    """
    2026-04-22 八次加固：把模型整段 raw_output 里的三段式 marker 命中情况抽出来。

    严格按用户规格使用字面量子串匹配（不做大小写/空白归一化）——训练端的 SFT target
    与评测端的匹配器共用同一套 marker 字面量（见 ``evaluation.metrics.MARKER_*`` 与
    ``dataset.adversarial.MARKER_*``），任何归一化都会让"部分匹配"混进真阳性，破坏
    ``full_compliance_rate`` 的语义。``raw_output=None`` 时稳定返回全 False，
    不抛错（None 本身属于"模型什么都没输出"这种已经被其他 FAIL FAST 覆盖的异常态）。
    """
    text = raw_output or ""
    return {
        "has_warning": MARKER_WARNING in text,
        "has_explanation": MARKER_EXPLANATION in text,
        "has_safe_solution": MARKER_SAFE_SOLUTION in text,
    }


def _truncate_prompt_leakage(model_output: str | None) -> str:
    """在抽取代码前移除泄漏的提示词段落。"""
    text = model_output or ""
    instruction_idx = text.find("### Instruction:")
    if instruction_idx != -1:
        text = text[:instruction_idx]
    input_idx = text.find("### Input:")
    if input_idx != -1:
        text = text[:input_idx]
    return text


def _append_invalid_extraction_debug(
    failures: list[dict[str, Any]],
    *,
    src: dict[str, Any],
    sample_id: int,
    raw_output: str | None,
    candidate: str | None,
    reason: str,
    source: str | None,
) -> None:
    print(f"[extract_python_code:invalid] sample_id={sample_id} id={src.get('id')!r}")
    print(f"[extract_python_code:invalid] reason={reason}")
    print(f"[extract_python_code:invalid] extracted_candidate={candidate!r}")
    print(f"[extract_python_code:invalid] raw_model_output={raw_output!r}")

    if len(failures) >= MAX_DEBUG_INVALID_SAMPLES:
        return
    failures.append(
        {
            "id": src.get("id"),
            "sample_index": sample_id,
            "expected_vulnerable": src.get("expected_vulnerable"),
            "attack_type": src.get("attack_type"),
            "difficulty": src.get("difficulty"),
            "task_type": src.get("task_type"),
            "source": source,
            "reason": reason,
            "extracted_candidate": candidate,
            "raw_model_output": raw_output,
        }
    )


def _write_invalid_extraction_debug(failures: list[dict[str, Any]]) -> None:
    DEBUG_INVALID_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_INVALID_SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(failures[:MAX_DEBUG_INVALID_SAMPLES], f, ensure_ascii=False, indent=2)
    print(
        f"[Eval] invalid extraction debug samples saved: "
        f"{min(len(failures), MAX_DEBUG_INVALID_SAMPLES)} -> {DEBUG_INVALID_SAMPLES_PATH}"
    )


def _assert_dataset_sanity(samples: list[dict[str, Any]]) -> tuple[int, int]:
    """
    评测启动前的数据集完整性门闸（FAIL FAST，2026-04-20 四次加固扩展版）。

    核心契约：三个 critical field 在**所有**样本上必须同时齐全且类型正确。任一样本
    违规即 ``RuntimeError`` 中断整次评测。严禁任何字段静默兜底。

      1) ``id``：非空字符串（与 ``_require_string_id`` 契约一致）。
      2) ``prompt``：非空字符串（``prompt_loader._normalize_sample`` 已构造）。
      3) ``expected_vulnerable``：Python ``bool``（``True`` 或 ``False``）。

    此外：
      - 样本列表不能为空。
      - 正类 / 负类样本数均须 > 0；只有一类会让 Precision/Recall/F1/FPR/FNR 数学上无效。
      - 打印启动日志：Total / Vulnerable / Safe，以及**有效标签百分比**
        ``[Eval] Valid labels: N/N (100.00%)``——由于缺失即 raise，这个百分比在
        成功启动时必为 100.00%；之所以仍打印出来，是让复现者一眼可见"本次运行有 100%
        的样本通过了 critical field 门闸"的审计痕迹。
    """
    if not samples:
        raise RuntimeError(
            "Invalid evaluation dataset: 样本数为 0，无法计算任何评测指标。"
        )

    total = len(samples)
    bad_fields: list[tuple[int, Any, str, str]] = []  # (idx, id, field, reason)
    for idx, s in enumerate(samples):
        sid_repr = repr(s.get("id"))
        for field_name in CRITICAL_SAMPLE_FIELDS:
            if field_name not in s:
                bad_fields.append((idx, s.get("id"), field_name, "missing"))

        if "id" in s and not (isinstance(s["id"], str) and s["id"].strip()):
            bad_fields.append(
                (idx, s.get("id"), "id",
                 f"must be non-empty str, got {type(s['id']).__name__}: {s['id']!r}")
            )
        if "prompt" in s and not (isinstance(s["prompt"], str) and s["prompt"].strip()):
            bad_fields.append(
                (idx, s.get("id"), "prompt",
                 f"must be non-empty str, got {type(s['prompt']).__name__}")
            )
        if "expected_vulnerable" in s and not isinstance(s["expected_vulnerable"], bool):
            bad_fields.append(
                (idx, s.get("id"), "expected_vulnerable",
                 f"must be bool, got {type(s['expected_vulnerable']).__name__}: "
                 f"{s['expected_vulnerable']!r}")
            )
        del sid_repr  # noqa: avoid unused-warning

    if bad_fields:
        # 先打印一次百分比，帮助用户定位数据质量比例（即使最终 raise）
        n_ok = total - len({idx for idx, _, _, _ in bad_fields})
        pct = 100.0 * n_ok / total if total else 0.0
        preview = "; ".join(
            f"#{idx}(id={sid!r}, field={fld}, reason={rsn})"
            for idx, sid, fld, rsn in bad_fields[:5]
        )
        more = f" ... and {len(bad_fields) - 5} more" if len(bad_fields) > 5 else ""
        raise RuntimeError(
            f"Invalid evaluation dataset: {len(bad_fields)} critical-field violations "
            f"across {total} samples (valid-label rate pre-raise: {n_ok}/{total}={pct:.2f}%). "
            f"violations: {preview}{more}"
        )

    n_pos = sum(1 for s in samples if s["expected_vulnerable"])
    n_neg = total - n_pos

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Invalid evaluation dataset: only one class present "
            f"(pos={n_pos}, neg={n_neg})"
        )

    pct = 100.0 * total / total  # 到这里必为 100.00%，显式打印以构成审计证据
    print(f"[Eval] Total samples: {total}")
    print(f"[Eval] Vulnerable: {n_pos}")
    print(f"[Eval] Safe: {n_neg}")
    print(
        f"[Eval] Valid labels: {total}/{total} ({pct:.2f}%) "
        f"[id+prompt+expected_vulnerable all present & well-typed]"
    )
    return n_pos, n_neg


def _require_expected_vulnerable(src: dict[str, Any]) -> bool:
    """从已加载的样本字典中严格读取 expected_vulnerable，不允许默认值回退。"""
    if "expected_vulnerable" not in src:
        raise KeyError(
            f"sample (id={src.get('id')!r}) 缺少 expected_vulnerable；"
            "prompt_loader 与数据集加载链应保证该字段存在。"
        )
    value = src["expected_vulnerable"]
    if not isinstance(value, bool):
        raise TypeError(
            f"sample (id={src.get('id')!r}) expected_vulnerable 类型必须是 bool，"
            f"实际为 {type(value).__name__}: {value!r}"
        )
    return value


def _require_string_id(src: dict[str, Any]) -> str:
    """严格读取样本 id，必须是非空字符串；禁止任何数字或默认值回退。

    评测数据使用不透明哈希 id（形如 ``sqlsec-<hex20>``），任何对 id 做数字假设
    （如 ``int(x["id"])``、按整数排序、把 dataloader 的位置索引当 id 回退）都会
    在第一条非数字 id 上直接崩溃或静默错位。这里在所有使用点做 FAIL FAST 校验，
    确保下游只看见字符串 id。
    """
    if "id" not in src:
        raise ValueError(
            "sample 缺少 id 字段；prompt_loader 与数据集构建链应保证该字段存在。"
        )
    sid = src["id"]
    if not isinstance(sid, str):
        raise ValueError(
            f"ID must be string; got type={type(sid).__name__} value={sid!r}"
        )
    if not sid.strip():
        raise ValueError(f"ID must be a non-empty string; got {sid!r}")
    return sid


def _per_sample_from_detection(
    det: dict[str, Any],
    *,
    src: dict[str, Any],
    sample_id: int,
    prompt: str,
    raw_output: str | None,
    code: str,
    invalid_extraction: bool,
    merge_mode: str,
    always_safe_stub: bool = False,
) -> dict[str, Any]:
    """仅用于 valid 抽取分支：det 必须是一次真实的 detect_vulnerability 输出。

    invalid 分支不走这里（走 ``_invalid_extraction_sample`` 构造 None 判定），
    因为 invalid 时根本没有可供 detector 判定的代码；硬调用 detector 会产生
    误导性的"全 False"预测，正是本次修复要铲除的漏洞来源。
    """
    if invalid_extraction:
        raise RuntimeError(
            "_per_sample_from_detection 不再接受 invalid_extraction=True 的调用；"
            "请改用 _invalid_extraction_sample 构造 is_vulnerable=None 的样本。"
        )
    bandit_block = det.get("bandit", {})
    issues = bandit_block.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    bandit_confidence_levels = [
        str(i.get("confidence", "UNKNOWN")).upper()
        for i in issues
        if isinstance(i, dict)
    ]
    b608 = bool(bandit_block.get("b608_hit"))
    bandit_any = bool(bandit_block.get("has_issue"))
    rb = det.get("rule_based", {})
    tt = det.get("taint", {})
    taint_on = not bool(tt.get("skipped", True))
    response_flags = _response_structure_flags(raw_output)
    return {
        "id": _require_string_id(src),
        "sample_index": sample_id,
        "prompt": prompt,
        "raw_output": raw_output,
        "code": code,
        "is_vulnerable": bool(det.get("is_vulnerable")),
        "attack_type": str(src.get("attack_type", "unknown")),
        "vulnerability_type": str(
            src.get("vulnerability_type", src.get("attack_type", "unknown"))
        ),
        "difficulty": str(src.get("difficulty", "unknown")),
        "task_type": str(src.get("task_type", "unknown")),
        "expected_vulnerable": _require_expected_vulnerable(src),
        "merge_mode": merge_mode,
        "detection_sources": list(det.get("detection_sources", [])),
        "bandit_issues": issues,
        "bandit_detected": bandit_any,
        "bandit_b608": b608,
        "bandit_issue_count": len(issues),
        "bandit_confidence_levels": bandit_confidence_levels,
        "bandit_has_B608": b608,
        "rule_based_detected": bool(rb.get("is_vulnerable")),
        "rule_based_violations": rb.get("violations", []),
        "taint_detected": bool(taint_on and tt.get("is_vulnerable")),
        "taint_flows_detected": int(tt.get("taint_flows_detected", 0)) if taint_on else 0,
        "fallback_violations": rb.get("violations", []),
        "invalid_extraction": False,
        "always_safe_stub": always_safe_stub,
        # 2026-04-22 八次加固：响应级结构 marker 命中情况（与 is_vulnerable 正交）。
        "has_warning": response_flags["has_warning"],
        "has_explanation": response_flags["has_explanation"],
        "has_safe_solution": response_flags["has_safe_solution"],
    }


def _invalid_extraction_sample(
    *,
    src: dict[str, Any],
    sample_id: int,
    prompt: str,
    raw_output: str | None,
    merge_mode: str,
) -> dict[str, Any]:
    """抽取失败分支：显式写 is_vulnerable=None，禁止与 valid 语义混淆。

    2026-04-21 修复 extraction-failure 漏洞：此前旧分支硬写 ``is_vulnerable=False``，
    把抽取失败的样本偷偷计入安全率。现在 invalid 样本的 is_vulnerable 字段值为 None，
    metrics 层会强制要求调用者先按 ``invalid_extraction`` 分流再读判定结果。
    任何把 None 读成 False 的代码会在 ``_require_is_vulnerable_respecting_invalid``
    处立即抛错，绝不允许静默兜底。

    2026-04-22 八次加固：invalid 分支同样写入 ``has_warning`` / ``has_explanation`` /
    ``has_safe_solution`` 三个响应级 marker 命中标志——即便 Python 代码抽取失败，模型的
    ``[SECURITY WARNING]`` / ``[EXPLANATION]`` 文本可能仍然完整（这两段是自然语言，不依赖
    Python 代码块），所以 invalid 样本**必须**参与响应质量指标的统计，否则
    ``warning_rate`` / ``explanation_rate`` 会漏掉一批真实存在的模型行为信号。
    """
    response_flags = _response_structure_flags(raw_output)
    return {
        "id": _require_string_id(src),
        "sample_index": sample_id,
        "prompt": prompt,
        "raw_output": raw_output,
        "code": "",
        "is_vulnerable": None,
        "attack_type": str(src.get("attack_type", "unknown")),
        "vulnerability_type": str(
            src.get("vulnerability_type", src.get("attack_type", "unknown"))
        ),
        "difficulty": str(src.get("difficulty", "unknown")),
        "task_type": str(src.get("task_type", "unknown")),
        "expected_vulnerable": _require_expected_vulnerable(src),
        "merge_mode": merge_mode,
        "detection_sources": [],
        "bandit_issues": [],
        "bandit_detected": False,
        "bandit_b608": False,
        "bandit_issue_count": 0,
        "bandit_confidence_levels": [],
        "bandit_has_B608": False,
        "rule_based_detected": False,
        "rule_based_violations": [],
        "taint_detected": False,
        "taint_flows_detected": 0,
        "fallback_violations": [],
        "invalid_extraction": True,
        "always_safe_stub": False,
        "has_warning": response_flags["has_warning"],
        "has_explanation": response_flags["has_explanation"],
        "has_safe_solution": response_flags["has_safe_solution"],
    }


def load_model_and_tokenizer(
    base_model: str,
    load_in_4bit: bool,
    adapter_path: str | None = None,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("评测阶段要求 CUDA GPU，禁止 CPU 回退。")

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quant,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    device = torch.device("cuda")
    try:
        model = model.to(device)
    except Exception:
        # 4bit + device_map="auto" 可能由 accelerate 管理设备分配，这里保持兼容。
        pass
    model.eval()
    return model, tok, device


def run_eval_always_safe(
    samples: list[dict[str, Any]],
    *,
    merge_mode: str = "or",
    enable_rule_based: bool = True,
    enable_taint: bool = False,
) -> MetricBundle:
    """
    基线健全性检查：不调用 LLM，每条样本恒返回同一参数化安全代码。
    预期：相对 expected_vulnerable 的 recall_vulnerable≈0（几乎抓不到「应为漏洞」类），
    用于验证标签与分类指标是否合理。
    """
    _assert_dataset_sanity(samples)
    prompts = [str(s["prompt"]) for s in samples]
    evaluated_samples: list[dict[str, Any]] = []
    code = ALWAYS_SAFE_SYNTHETIC_OUTPUT.strip()
    text = code

    for sample_id, src in enumerate(samples):
        dres = detect_vulnerability(
            code,
            sample_id=sample_id,
            merge_mode=merge_mode,  # type: ignore[arg-type]
            enable_rule_based=enable_rule_based,
            enable_taint=enable_taint,
        )
        evaluated_samples.append(
            _per_sample_from_detection(
                dres,
                src=src,
                sample_id=sample_id,
                prompt=prompts[sample_id],
                raw_output=text,
                code=code,
                invalid_extraction=False,
                merge_mode=merge_mode,
                always_safe_stub=True,
            )
        )

    bundle = aggregate_metrics(evaluated_samples)
    print_eval_summary(bundle)
    return bundle


def run_eval_on_prompts(
    samples: list[dict[str, Any]],
    base_model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool,
    adapter_path: str | None = None,
    per_device_eval_batch_size: int = 4,
    dataloader_num_workers: int = 2,
    dataloader_pin_memory: bool = True,
    debug_timing: bool = True,
    merge_mode: str = "or",
    enable_rule_based: bool = True,
    enable_taint: bool = False,
) -> MetricBundle:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    _assert_dataset_sanity(samples)
    model, tok, device = load_model_and_tokenizer(
        base_model=base_model,
        load_in_4bit=load_in_4bit,
        adapter_path=adapter_path,
    )

    source_samples = samples
    prompts = [str(s["prompt"]) for s in source_samples]
    enc = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.arange(len(prompts), dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(per_device_eval_batch_size)),
        shuffle=False,
        num_workers=max(0, int(dataloader_num_workers)),
        pin_memory=bool(dataloader_pin_memory),
    )

    print(f"[eval] batch_size={max(1, int(per_device_eval_batch_size))}")
    print(f"[eval] device={device}")
    print(
        f"[eval] dataloader workers={max(0, int(dataloader_num_workers))}, "
        f"pin_memory={bool(dataloader_pin_memory)}"
    )

    evaluated_samples: list[dict[str, Any]] = []
    invalid_debug_samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, sample_ids) in enumerate(tqdm(loader, desc="eval")):
            t0 = time.perf_counter()
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if batch_idx == 0:
                print(f"[eval] input tensor device={input_ids.device}")

            do_sample = temperature > 0
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            for row in range(out_ids.size(0)):
                prompt_len = int(attention_mask[row].sum().item())
                gen_tokens = out_ids[row, prompt_len:]
                text = tok.decode(gen_tokens, skip_special_tokens=True)
                cleaned_for_extraction = _truncate_prompt_leakage(text)
                extraction = extract_python_code_with_debug(cleaned_for_extraction)
                code = extraction.code
                sample_id = int(sample_ids[row].item())
                src = source_samples[sample_id]

                if code is None:
                    _append_invalid_extraction_debug(
                        invalid_debug_samples,
                        src=src,
                        sample_id=sample_id,
                        raw_output=text,
                        candidate=extraction.candidate,
                        reason=extraction.reason,
                        source=extraction.source,
                    )
                    evaluated_samples.append(
                        _invalid_extraction_sample(
                            src=src,
                            sample_id=sample_id,
                            prompt=prompts[sample_id],
                            raw_output=text,
                            merge_mode=merge_mode,
                        )
                    )
                    continue

                dres = detect_vulnerability(
                    code,
                    sample_id=sample_id,
                    merge_mode=merge_mode,  # type: ignore[arg-type]
                    enable_rule_based=enable_rule_based,
                    enable_taint=enable_taint,
                )
                evaluated_samples.append(
                    _per_sample_from_detection(
                        dres,
                        src=src,
                        sample_id=sample_id,
                        prompt=prompts[sample_id],
                        raw_output=text,
                        code=code,
                        invalid_extraction=False,
                        merge_mode=merge_mode,
                    )
                )

            if debug_timing:
                dt = time.perf_counter() - t0
                print(f"[eval] batch={batch_idx} time={dt:.3f}s")

    for sample in evaluated_samples:
        if not isinstance(sample["id"], str):
            raise ValueError(
                f"ID must be string; got type={type(sample['id']).__name__} "
                f"value={sample['id']!r}"
            )
    evaluated_samples.sort(key=lambda x: x["id"])

    # 评测循环结束后立刻打印 Invalid 样本数与 extraction failure 率，方便在指标
    # 聚合 raise（extraction_failure_rate > 0.5）之前留下可审计日志。
    n_invalid_preview = sum(
        1 for s in evaluated_samples if s.get("invalid_extraction") is True
    )
    n_total_preview = len(evaluated_samples)
    preview_rate = n_invalid_preview / n_total_preview if n_total_preview > 0 else 0.0
    print(
        f"[Eval] extraction summary: invalid={n_invalid_preview}/{n_total_preview} "
        f"(rate={preview_rate:.4f}); aggregate_metrics 会在 rate>0.5 时 RuntimeError。"
    )
    _write_invalid_extraction_debug(invalid_debug_samples)

    bundle = aggregate_metrics(evaluated_samples)
    print_eval_summary(bundle)
    return bundle


def save_results(path: Path, bundle: MetricBundle, meta: dict[str, Any]) -> None:
    """把 invalid-extraction 语义加固版的 MetricBundle 写入评测 JSON。

    ``summary`` 顶层严格使用 ``_valid`` 后缀或 ``valid_only_metrics`` /
    ``conservative_metrics`` / ``strict_metrics`` 三个 bundle 承载；旧的
    ``overall_sql_injection_rate`` / ``sql_injection_rate`` /
    ``safe_code_generation_rate`` / ``classification_vs_expected`` 这些
    "混入 invalid 样本" 的字段已被彻底移除（破坏性变更，不提供兼容层）。

    2026-04-22 八次加固：额外写入 ``response_quality_metrics``（响应级 3 段式合规率）。
    该字段是**纯新增**，既有字段的键名、类型、数值语义**完全不变**，因此下游
    ``scripts/compare_results.py::load_summary`` 的必填字段校验依然通过，旧评测 JSON
    继续被拒（没有 ``response_quality_metrics`` 的新 JSON 也是可读的——下游脚本只会在
    新字段不存在时给出默认展示，不会 raise，见该文件的兼容处理）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": {
            "n_samples": bundle.n_samples,
            "n_valid": bundle.n_valid,
            "n_invalid": bundle.n_invalid,
            "extraction_failure_rate": bundle.extraction_failure_rate,
            "sql_injection_rate_valid": bundle.sql_injection_rate_valid,
            "safe_rate_valid": bundle.safe_rate_valid,
            "valid_only_metrics": bundle.valid_only_metrics,
            "conservative_metrics": bundle.conservative_metrics,
            "strict_metrics": bundle.strict_metrics,
            "bandit_total_detections": bundle.bandit_total_detections,
            "bandit_detection_rate": bundle.bandit_detection_rate,
            "bandit_b608_rate": bundle.bandit_b608_rate,
            "bandit_low_confidence_count": bundle.bandit_low_confidence_count,
            "bandit_medium_confidence_count": bundle.bandit_medium_confidence_count,
            "bandit_high_confidence_count": bundle.bandit_high_confidence_count,
            "bandit_confidence_distribution": bundle.bandit_confidence_distribution,
            "bandit_risk_score": bundle.bandit_risk_score,
            "b608_detection_rate": bundle.b608_detection_rate,
            "by_attack_type_valid": bundle.by_attack_type_valid,
            "by_difficulty_valid": bundle.by_difficulty_valid,
            "by_task_type_valid": bundle.by_task_type_valid,
            "detection_layer_stats": bundle.detection_layer_stats,
            "detection_source_breakdown": bundle.detection_source_breakdown,
            "per_detector_vs_expected": bundle.per_detector_vs_expected,
            "by_attack_type_metrics": bundle.by_attack_type_metrics,
            "response_quality_metrics": bundle.response_quality_metrics,
        },
        "per_sample": bundle.per_sample,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
