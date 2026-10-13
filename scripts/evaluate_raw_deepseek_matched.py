"""Real DeepSeek rejudgements on the immutable EEG evidence of a saved GPT run.

Prediction never reads the scoring CSV or saved GPT classes. Successful cloud
decisions are immutable; only failed decisions receive up to two recovery tries.
"""
import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_agent import raw_tool_agent
from eeg_agent.agents import CloudProvider
from eeg_agent.cloud_config import read_settings
from eeg_agent.config import file_hash, load_config
from scripts.evaluate_raw_gpt_all import read, save, score_rows


def now():
    return datetime.now(timezone.utc).isoformat()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def prepare(source, out):
    if out.exists():
        raise ValueError("Use a fresh output directory")
    original = read(source / "protocol.json")
    if original["total"] != 146 or original["subject_count"] != 24:
        raise ValueError("Unexpected GPT comparison denominator")
    spec = importlib.util.spec_from_file_location("frozen_gpt_agent", source / "sources/eeg_agent_raw_tool_agent.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if (frozen.TOOLS != raw_tool_agent.TOOLS or frozen.SCHEMA != raw_tool_agent.SCHEMA
            or frozen.SYSTEM.replace("the GPT EEG", "the cloud EEG") != raw_tool_agent.SYSTEM):
        raise ValueError("Decision policy differs from the GPT benchmark")
    settings = read_settings()
    if settings["EEG_API_MODEL"] != "deepseek-flash" or settings["EEG_API_BASE_URL"].rstrip("/") != "https://api.deepseek.com":
        raise ValueError("Use the configured native DeepSeek Flash profile")
    if not settings["EEG_API_KEY"]:
        raise ValueError("Missing project DeepSeek key")
    inputs = []
    for subject in sorted({p["subject"] for p in original["paths"]}):
        saved = read(source / "predictions" / f"subject_{subject:02d}.json")
        result = saved["result"]
        if not result["fresh_raw_encoding"] or result["signal_sha256"] != original["subjects"][str(subject)]["raw_sha256"]:
            raise ValueError("Wrong EEG provenance")
        segments = {r["ordinal"]: r for r in result["measurements"]["segments"]}
        for path in [p for p in original["paths"] if p["subject"] == subject]:
            segment = segments[path["ordinal"]]
            if (not segment["complete"] or segment["fresh_encoded_windows"] != segment["accepted_windows"]
                    or segment["numeric_fusion"]["reference_available"]
                    or abs(segment["start_seconds"] * 1024 - path["raw_start_sample"]) > 1e-6
                    or abs(segment["end_seconds"] * 1024 - path["raw_end_sample"]) > 1e-6):
                raise ValueError("Wrong whole-path evidence")
            prediction = {key: copy.deepcopy(segment[key]) for key in
                          ("mil_raw_high_probability", "accepted_windows", "numeric_fusion")}
            runtime = raw_tool_agent.EvidenceRuntime(prediction)
            expected_class = "High" if prediction["numeric_fusion"]["raw_combined_score"] >= .5 else "Low"
            if prediction["numeric_fusion"]["predicted_class"] != expected_class:
                raise ValueError("Invalid frozen numerical class")
            runtime.execute("corrective_evidence")
            inputs.append({**path, "prediction": prediction, "evidence_sha256": value_hash(prediction)})
    if len(inputs) != 146 or len({p["path_key"] for p in inputs}) != 146:
        raise ValueError("Incomplete fixed evidence")
    original_files = [p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    frozen_files = [Path(__file__), ROOT / "eeg_agent/raw_tool_agent.py", ROOT / "eeg_agent/agents.py",
                    ROOT / "eeg_agent/cloud_config.py", ROOT / "eeg_agent/config.py",
                    ROOT / "scripts/evaluate_raw_gpt_all.py", ROOT / "configs/local.yaml", ROOT / ".env",
                    ROOT / "cloud_profiles.json"]
    out.mkdir(parents=True)
    save(out / "frozen_inputs.json", inputs)
    protocol = {"schema": "raw_eeg_deepseek_matched_v1", "created_at": now(), "gpt_source": str(source),
                "provider_model": settings["EEG_API_MODEL"], "base_url": settings["EEG_API_BASE_URL"],
                "total": 146, "subjects": 24, "model_seed": 2026, "label_threshold": 30,
                "label_file": original["label_file"], "label_sha256": original["label_sha256"],
                "input_sha256": file_hash(out / "frozen_inputs.json"),
                "original_file_sha256": {str(p.relative_to(source)): file_hash(p) for p in original_files},
                "source_sha256": {str(p): file_hash(p) for p in frozen_files},
                "numerical_evidence": "exact frozen local results from the GPT raw-EEG evaluation",
                "fresh_raw_encoding_this_run": False, "fresh_cloud_classification": True,
                "numerical_fitting_calls": 0, "gpt_classes_sent_to_deepseek": False,
                "target_labels_sent_to_deepseek": False, "baseline_policy": "disabled",
                "api_timeout_seconds": 60, "api_max_calls_per_subject": 50, "workers_per_subject": 3,
                "maximum_recovery_attempts_per_path": 2, "consecutive_recovery_failure_circuit_break": 3,
                "failure_policy": "never retry valid decisions; failures count wrong with fixed denominator 146",
                "cloud_settings": {"DeepSeek": "native Chat Completions; temperature=0; thinking disabled",
                                   "GPT": "saved Responses run; reasoning effort medium"},
                "prompt_policy_match": True,
                "prompt_provenance_change": "truthfully describe immutable replay rather than new signal computation"}
    save(out / "protocol.json", protocol)
    for path in frozen_files:
        if path.suffix == ".py":
            target = out / "sources" / (path.parent.name + "_" + path.name)
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(path.read_bytes())
    save(out / "progress.json", {"stage": "prepared", "total": 146, "completed_paths": 0})
    print(json.dumps({"stage": "prepared", "paths": 146, "model": protocol["provider_model"],
                      "evidence_reused": True, "labels_used": False}), flush=True)


def verify(out):
    p = read(out / "protocol.json")
    source = Path(p["gpt_source"])
    for name, expected in p["original_file_sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError("Original GPT artifact changed")
    for name, expected in p["source_sha256"].items():
        if file_hash(name) != expected:
            raise ValueError("Frozen evaluation source or project configuration changed")
    if file_hash(out / "frozen_inputs.json") != p["input_sha256"]:
        raise ValueError("Frozen input changed")
    return p, read(out / "frozen_inputs.json")


class ReplayRuntime(raw_tool_agent.EvidenceRuntime):
    def initial_evidence(self):
        value = super().initial_evidence()
        value["numerical_execution"] = "replayed_frozen_raw_eeg_evidence; callable immutable evidence"
        return value

    def execute(self, name):
        value = super().execute(name)
        if name == "corrective_evidence":
            value["scope"] = "whole-path; same frozen raw EEG evidence; no target or resting reference"
            self.memo[name]["scope"] = value["scope"]
        return value


class AuditedProvider(CloudProvider):
    def __init__(self, cfg, out):
        self.out = out
        super().__init__("api", cfg, lambda kind, payload: None)

    def call(self, role, payload, tools=None, messages=None):
        request = {"created_at": now(), "role": role, "payload": payload,
                   "tools": copy.deepcopy(tools), "messages": copy.deepcopy(messages),
                   "model": self.model, "simulated": False}
        try:
            response = super().call(role, payload, tools=tools, messages=messages)
            request.update(response_id=response.get("id"), response_model=response.get("model"))
            return response
        except Exception as exc:
            request["error_type"] = type(exc).__name__
            raise
        finally:
            request["finished_at"] = now()
            save(self.out / "requests" / (uuid.uuid4().hex + ".json"), request)


def attempts(out, path):
    return [read(p) for p in sorted((out / "attempts").glob(path["path_key"].replace(":", "_") + "_*.json"))]


def chosen(saved):
    return next((a for a in saved if a["judgment"]["available"]), saved[-1] if saved else None)


def refresh(out, inputs, stage):
    rows = []
    for path in inputs:
        saved = attempts(out, path)
        decision = chosen(saved)
        judgment = decision["judgment"] if decision else {}
        rows.append({k: path[k] for k in path if k != "prediction"} | {
            "final_class": judgment.get("final_class") if judgment.get("available") else None,
            "generation_status": judgment.get("generation_status", "pending"),
            "confidence": judgment.get("confidence"), "explanation": judgment.get("explanation", ""),
            "actual_tool_calls": judgment.get("actual_tool_calls", []),
            "attempt_count": len(saved), "selected_attempt": decision["attempt"] if decision else None})
    usage = [read(p) for p in (out / "usage").glob("*.json")]
    progress = {"stage": stage, "total": 146, "completed_paths": sum(r["attempt_count"] > 0 for r in rows),
                "valid_cloud_paths": sum(r["final_class"] is not None for r in rows),
                "attempts": sum(r["attempt_count"] for r in rows),
                "api_calls": sum(p["calls"] for p in usage), "api_failed_calls": sum(p["failures"] for p in usage),
                "input_tokens": sum(p["input_tokens"] for p in usage),
                "output_tokens": sum(p["output_tokens"] for p in usage),
                "fresh_encoded_windows_this_run": 0, "reused_eeg_windows": sum(p["prediction"]["accepted_windows"] for p in inputs),
                "labels_used_for_prediction": False, "updated_at": now()}
    save(out / "unscored_predictions.json", rows)
    save(out / "progress.json", progress)
    return rows, progress


def predict(out):
    protocol, inputs = verify(out)
    if (out / "prediction_complete.json").exists():
        raise ValueError("Predictions already completed; successful decisions must not be rerun")
    raw_tool_agent.EvidenceRuntime = ReplayRuntime
    raw_tool_agent.SYSTEM = raw_tool_agent.SYSTEM.replace(
        "The numerical evidence was freshly recomputed from raw EEG before this cloud\nrequest; callable tools read those immutable local results.",
        "The numerical evidence is the immutable raw EEG evidence from a fixed prior\nlocal computation; callable tools read those same frozen local results.")
    raw_tool_agent.TOOLS = copy.deepcopy(raw_tool_agent.TOOLS)
    for tool in raw_tool_agent.TOOLS:
        tool["description"] = tool["description"].replace("freshly computed EEG", "fixed EEG")
    cfg = load_config()
    cfg["api_timeout_seconds"] = 60
    cfg["api_max_calls"] = 50

    def run_one(path, attempt, provider):
        target = out / "attempts" / (path["path_key"].replace(":", "_") + f"_{attempt}.json")
        if target.exists() or any(a["judgment"]["available"] for a in attempts(out, path)):
            raise ValueError("Attempt overwrite or retry of a successful decision")
        judgment = raw_tool_agent.classify(provider, copy.deepcopy(path["prediction"]))
        judgment["numerical_execution"] = "replayed_frozen_raw_eeg_evidence"
        save(target, {"path_key": path["path_key"], "attempt": attempt, "finished_at": now(),
                      "evidence_sha256": path["evidence_sha256"], "judgment": judgment, "labels_used": False})
        print(json.dumps({"stage": "first_pass" if attempt == 0 else "recovery",
                          "path": path["path_key"], "attempt": attempt, "available": judgment["available"],
                          "error_type": judgment.get("error_type")}), flush=True)
        return judgment

    for subject in sorted({p["subject"] for p in inputs}):
        pending = [p for p in inputs if p["subject"] == subject and not attempts(out, p)]
        if not pending:
            continue
        provider = AuditedProvider(cfg, out)
        if provider.model != protocol["provider_model"]:
            raise ValueError("DeepSeek provider changed")
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(run_one, path, 0, provider) for path in pending]
                for future in as_completed(futures):
                    future.result()
            save(out / "usage" / f"first_subject_{subject:02d}_{uuid.uuid4().hex[:8]}.json", provider.summary())
        finally:
            provider.client.close()
        _, progress = refresh(out, inputs, "first_pass")
        print(json.dumps(progress), flush=True)
    first_rows, first_progress = refresh(out, inputs, "first_pass_complete")
    save(out / "first_pass_predictions.json", first_rows)
    save(out / "first_pass_complete.json", first_progress)
    for retry in (1, 2):
        consecutive_failures = 0
        for path in inputs:
            saved = attempts(out, path)
            if chosen(saved)["judgment"]["available"] or len(saved) != retry:
                continue
            provider = AuditedProvider(cfg, out)
            try:
                judgment = run_one(path, retry, provider)
                save(out / "usage" / (path["path_key"].replace(":", "_") + f"_recovery_{retry}.json"), provider.summary())
            finally:
                provider.client.close()
            consecutive_failures = 0 if judgment["available"] else consecutive_failures + 1
            refresh(out, inputs, "recovery")
            if consecutive_failures >= 3:
                print(json.dumps({"stage": "recovery_circuit_break", "pass": retry}), flush=True)
                break
    verify(out)
    rows, progress = refresh(out, inputs, "prediction_complete")
    if progress["completed_paths"] != 146:
        raise ValueError("Incomplete denominator")
    save(out / "prediction_complete.json", progress)
    print(json.dumps(progress), flush=True)


def score(out):
    protocol, inputs = verify(out)
    complete = read(out / "prediction_complete.json")
    predictions = read(out / "unscored_predictions.json")
    if complete["completed_paths"] != 146 or len(predictions) != 146:
        raise ValueError("Do not score an incomplete evaluation")
    if file_hash(protocol["label_file"]) != protocol["label_sha256"]:
        raise ValueError("Scoring labels changed")
    # First read of query truth/class values in this runner.
    truths = {}
    with Path(protocol["label_file"]).open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["path_score_available"].lower() == "true" and r["path_score"]:
                key = f"{int(r['subject_id']):02d}:{int(r['start_sample_1024'])}"
                value = float(r["path_score"])
                truths[key] = {"path_score": value, "true_class": "High" if value >= 30 else "Low"}

    def scored(values):
        rows, metrics = score_rows([{**r, "gpt_class": r["final_class"]} for r in values], truths)
        metrics["valid_cloud_paths"] = metrics.pop("valid_gpt_paths")
        return [{k: v for k, v in r.items() if k != "gpt_class"} for r in rows], metrics

    rows, metrics = scored(predictions)
    _, first_metrics = scored(read(out / "first_pass_predictions.json"))
    with (Path(protocol["gpt_source"]) / "path_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        gpt = {r["path_key"]: r for r in csv.DictReader(stream)}
    if set(gpt) != {r["path_key"] for r in rows}:
        raise ValueError("GPT and DeepSeek path sets differ")
    gpt_correct = 0
    for row in rows:
        baseline = gpt[row["path_key"]]
        if row["true_class"] != baseline["true_class"] or row["path_score"] != float(baseline["path_score"]):
            raise ValueError("Scoring labels differ")
        row["deepseek_class"] = row.pop("final_class")
        row["gpt_class"] = baseline["gpt_class"]
        row["gpt_correct"] = row["gpt_class"] == row["true_class"]
        row["deepseek_correct"] = row.pop("correct")
        row["same_decision"] = row["deepseek_class"] == row["gpt_class"]
        gpt_correct += row["gpt_correct"]
    if gpt_correct != 108:
        raise ValueError("Unexpected saved GPT reference")
    paired = {"same_decisions": sum(r["same_decision"] for r in rows),
              "different_decisions": sum(not r["same_decision"] for r in rows),
              "deepseek_correct_gpt_wrong": sum(r["deepseek_correct"] and not r["gpt_correct"] for r in rows),
              "gpt_correct_deepseek_wrong": sum(r["gpt_correct"] and not r["deepseek_correct"] for r in rows),
              "both_correct": sum(r["gpt_correct"] and r["deepseek_correct"] for r in rows),
              "both_wrong": sum(not r["gpt_correct"] and not r["deepseek_correct"] for r in rows)}
    with (out / "path_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "actual_tool_calls": ";".join(r["actual_tool_calls"])} for r in rows)
    summary = {"status": "completed_and_scored", "protocol": protocol["schema"],
               "deepseek_model": protocol["provider_model"], "deepseek_metrics": metrics,
               "first_pass_metrics": first_metrics, "gpt_model": "gpt-6.1-sol",
               "gpt_metrics": {"correct": gpt_correct, "total": 146, "accuracy": gpt_correct / 146},
               "acc_difference_percentage_points": (metrics["accuracy"] - gpt_correct / 146) * 100,
               "paired": paired, "execution": complete, "scored_at": now(),
               "limitations": ["one configured API run per provider; no repeated-run stability estimate",
                               "local dataset repeatedly used in development; no external generalization",
                               "same numerical evidence/policy, provider-specific API and reasoning settings",
                               "exact EEG evidence reused; DeepSeek classes newly requested, no retraining"]}
    save(out / "summary.json", summary)
    report = (
        "# DeepSeek 与 GPT 原始 EEG 分类对照\n\n"
        f"| 模型 | 正确/总数 | ACC | 有效类别 |\n|---|---:|---:|---:|\n"
        f"| DeepSeek ({protocol['provider_model']}) | {metrics['correct']}/146 | {metrics['accuracy']*100:.2f}% | {metrics['valid_cloud_paths']}/146 |\n"
        f"| GPT (gpt-6.1-sol，已保存对照) | {gpt_correct}/146 | {gpt_correct/146*100:.2f}% | 146/146 |\n\n"
        f"DeepSeek − GPT：{summary['acc_difference_percentage_points']:+.2f} 个百分点；"
        f"同类判断 {paired['same_decisions']} 条，不同判断 {paired['different_decisions']} 条。\n\n"
        f"DeepSeek 判断正确而 GPT 错误 {paired['deepseek_correct_gpt_wrong']} 条；"
        f"GPT 正确而 DeepSeek 错误 {paired['gpt_correct_deepseek_wrong']} 条。\n\n"
        f"真实 DeepSeek API 请求 {complete['api_calls']} 次，其中失败 {complete['api_failed_calls']} 次。"
        f"初次分类 ACC {first_metrics['accuracy']*100:.2f}%（{first_metrics['correct']}/146），"
        f"有效类别 {first_metrics['valid_cloud_paths']}/146。失败分类最多补跑两次，成功类别不重试。"
        f"最终覆盖率 {metrics['coverage']*100:.2f}%；失败仍按错误计入固定分母 146。\n\n"
        f"混淆矩阵（行真实 Low/High，列预测 Low/High）：{metrics['confusion_matrix']}；"
        f"BACC {metrics['balanced_accuracy']*100:.2f}%。\n\n"
        "本次复用 GPT 原始 EEG 全路径评测中冻结的本地数值证据；没有重新编码 EEG，"
        "没有训练或更改权重。DeepSeek 最终类别和文字由本次官网 API 新生成。"
        "预测请求不传 GPT 类别或目标标签；真实标签只在全部预测冻结后的评分阶段读取。"
        "分类决策提示与原 GPT 评测相同，仅使用中性模型称呼并如实标明证据复用来源；"
        "DeepSeek 通过原生 Chat Completions 调用，temperature=0，思考模式关闭，"
        "GPT 对照来自先前 Responses medium 推理设置。因此比较的是这两种配置下的工具 Agent。"
        "两者均以既有本地 MIL 和八路数值工具作为依据，不能解释为直接读取原始波形的独立分类能力。\n\n"
        "这是同一开发数据上的单次对照，不证明某模型稳定优于另一个，也不代表外部泛化或临床诊断准确率。\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "predict", "score"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.source.resolve(), args.out.resolve())
    else:
        {"predict": predict, "score": score}[args.stage](args.out.resolve())
