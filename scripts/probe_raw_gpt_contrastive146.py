"""Run CPA GPT high-reasoning over the audited optimized RAG 146-path set."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

APP = Path(__file__).resolve().parents[1]
RESEARCH = Path(r"D:\桌面\文献总\EEGAGENT")
sys.path[:0] = [str(APP), str(RESEARCH)]

from eeg_agent import cloud_config
from openai import OpenAI
from scripts import probe_raw_deepseek_thinking43 as deep

OUT = RESEARCH / "outputs/vrms_rag_probes/20261011_gpt_high_contrastive146_01"
MANIFEST = RESEARCH / "outputs/vrms_rag_probes/current_raw_contrastive_all146_contexts_20261011_v2_cpa_gpt/arm_contexts.json"
MODEL = "gpt-6.1-sol"
MAX_OUTPUT_TOKENS = 8192
TIMEOUT = 180
WORKERS = 4

FINAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "final_class": {"type": "string", "enum": ["High", "Low"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "explanation": {"type": "string"},
        "supporting_tools": {"type": "array", "items": {"type": "string"}},
        "counterevidence_tools": {"type": "array", "items": {"type": "string"}},
        "rag_citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "final_class", "confidence", "explanation", "supporting_tools",
        "counterevidence_tools", "rag_citations",
    ],
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def citation_ids(value):
    ids = set()
    if isinstance(value, dict):
        ids.update(value[k] for k in ("citation_id", "case_id")
                   if isinstance(value.get(k), str))
        for nested in value.values():
            ids.update(citation_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            ids.update(citation_ids(nested))
    return ids


def validate_final(value, payload):
    fields = set(FINAL_SCHEMA["required"])
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid six-field final")
    if value["final_class"] not in ("High", "Low"):
        raise ValueError("invalid final class")
    if value["confidence"] not in ("low", "medium", "high"):
        raise ValueError("invalid confidence")
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise ValueError("missing explanation")
    for key in ("supporting_tools", "counterevidence_tools", "rag_citations"):
        if not isinstance(value[key], list) or any(not isinstance(x, str) for x in value[key]):
            raise ValueError("invalid evidence array")
    allowed_tools = {"MIL", *deep.base.raw_tool_agent.TOOL_NAMES}
    if not set(value["supporting_tools"] + value["counterevidence_tools"]).issubset(allowed_tools):
        raise ValueError("unknown tool citation")
    if not set(value["rag_citations"]).issubset(citation_ids(payload)):
        raise ValueError("unknown RAG citation")
    return value


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing output: {OUT}")
    source, rows = deep.selection("all146")
    contexts, hashes = deep.context_manifest(MANIFEST, "all146", "contrastive_rag", rows)
    predictions = {}
    payloads = {}
    teaching = deep.base.read(deep.TEN_SOURCE / "../" / "teaching.json") if False else deep.base.build_teaching_context()
    for row in rows:
        prediction, _, _ = deep.load_prediction(row)
        predictions[row["query_id"]] = prediction
        payload = deep.build_payload(prediction, teaching, "contrastive_rag", row,
                                     contexts[row["query_id"]])
        payloads[row["query_id"]] = payload
    OUT.mkdir(parents=True)
    save(OUT / "selection.json", rows)
    save(OUT / "payloads.json", payloads)
    save(OUT / "frozen_predictions.json", predictions)
    save(OUT / "protocol.json", {
        "schema": "cpa_gpt_high_contrastive_rag146_v1",
        "created_at": stamp(), "scope": "all146", "fixed_total": 146,
        "selection_source": str(source), "selection_sha256": sha(source),
        "model": MODEL, "profile_id": "cpa_gpt", "reasoning_effort": "high",
        "max_output_tokens": MAX_OUTPUT_TOKENS, "timeout_seconds": TIMEOUT,
        "max_retries": 0, "requests_per_judgment": 1, "workers": WORKERS,
        "rag_manifest": str(MANIFEST), "rag_manifest_sha256": sha(MANIFEST),
        "rag_source_hashes": hashes, "final_decision_source": "LLM_only",
        "query_truth_sent": False, "previous_cloud_answer_sent": False,
        "fusion_class_score_sent": False,
    })
    print(json.dumps({"prepared": 146, "manifest": str(MANIFEST), "api_requests": 0}, ensure_ascii=False))


def run_one(row, payload, profile, allow_recovery=False):
    query_id = row["query_id"]
    judgment_dir = OUT / "judgments" / query_id
    judgment_dir.mkdir(parents=True, exist_ok=True)
    existing = judgment_dir / "summary.json"
    if (judgment_dir / "attempt.json").exists() and not allow_recovery:
        return {"query_id": query_id, "status": "already_attempted"}
    if allow_recovery and (not existing.exists() or read(existing).get("status") != "failed_without_retry"):
        return {"query_id": query_id, "status": "not_recoverable"}
    save(judgment_dir / ("recovery_attempt.json" if allow_recovery else "attempt.json"),
         {"started_at": stamp(), "model": MODEL, "recovery": allow_recovery})
    client = OpenAI(base_url=profile["base_url"], api_key=profile["api_key"],
                    timeout=TIMEOUT, max_retries=0)
    messages = [
        {"role": "system", "content": deep.SYSTEM +
         "\nThis is the CPA GPT high-reasoning run. Return exactly the six required fields and no extra fields."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                  separators=(",", ":"), allow_nan=False)},
    ]
    request = {
        "model": MODEL, "input": messages, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False, "reasoning": {"effort": "high"},
        "text": {"format": {"type": "json_schema", "name": "cpa_vrms_final",
                               "schema": FINAL_SCHEMA, "strict": True}},
    }
    save(judgment_dir / "request.json", request)
    try:
        response = client.responses.create(**request)
        native = response.model_dump(mode="json")
        save(judgment_dir / "private_response.json", native)
        text = getattr(response, "output_text", None)
        if not text:
            raise ValueError("missing output_text")
        final = validate_final(json.loads(text), payload)
        save(judgment_dir / "judgment.json", final)
        result = {"query_id": query_id, "status": "cloud_verified",
                  "final_class": final["final_class"], "confidence": final["confidence"]}
        save(judgment_dir / "summary.json", result)
        return result
    except Exception as exc:
        result = {"query_id": query_id, "status": "failed_without_retry",
                  "error_type": type(exc).__name__, "final_class": None}
        save(judgment_dir / "summary.json", result)
        return result


def score():
    rows = read(OUT / "selection.json")
    result_rows = []
    for row in rows:
        summary = read(OUT / "judgments" / row["query_id"] / "summary.json")
        final_class = summary.get("final_class")
        true_class = row["true_class"]
        result_rows.append({**row, "new_class": final_class,
                            "valid": final_class in ("High", "Low"),
                            "correct": final_class == true_class})
    valid = [r for r in result_rows if r["valid"]]
    correct = sum(r["correct"] for r in result_rows)
    valid_correct = sum(r["correct"] for r in valid)
    high_total = sum(r["true_class"] == "High" for r in result_rows)
    low_total = sum(r["true_class"] == "Low" for r in result_rows)
    high_correct = sum(r["correct"] and r["true_class"] == "High" for r in result_rows)
    low_correct = sum(r["correct"] and r["true_class"] == "Low" for r in result_rows)
    summary = {
        "schema": "cpa_gpt_high_contrastive_rag146_results_v1",
        "scope": "all146", "fixed_total": 146,
        "valid": len(valid), "correct_fixed_denominator": correct,
        "accuracy_fixed_denominator": correct / len(result_rows),
        "accuracy_valid_only": valid_correct / len(valid) if valid else 0,
        "class_recall_fixed_denominator": {
            "High": high_correct / high_total if high_total else 0,
            "Low": low_correct / low_total if low_total else 0,
        },
        "balanced_accuracy_fixed_denominator": ((high_correct / high_total) +
                                                  (low_correct / low_total)) / 2,
        "failed_or_invalid": len(result_rows) - len(valid),
        "path_results": result_rows,
    }
    save(OUT / "summary.json", summary)
    with (OUT / "path_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(result_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "path_results"}, ensure_ascii=False))


def run():
    profile = cloud_config._profile_state(None)["profiles"]["cpa_gpt"]
    if profile["model_id"] != MODEL or not profile["api_key"]:
        raise ValueError("CPA GPT profile is not configured for gpt-6.1-sol")
    rows = read(OUT / "selection.json")
    payloads = read(OUT / "payloads.json")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, row, payloads[row["query_id"]], profile)
                   for row in rows]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
    score()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "run", "recover", "score"):
        raise SystemExit("usage: prepare|run|recover|score")
    if sys.argv[1] == "recover":
        profile = cloud_config._profile_state(None)["profiles"]["cpa_gpt"]
        rows = read(OUT / "selection.json")
        payloads = read(OUT / "payloads.json")
        failed = [row for row in rows
                  if (OUT / "judgments" / row["query_id"] / "summary.json").exists()
                  and read(OUT / "judgments" / row["query_id"] / "summary.json").get("status") == "failed_without_retry"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(run_one, row, payloads[row["query_id"]], profile, True) for row in failed]
            for future in as_completed(futures):
                print(json.dumps(future.result(), ensure_ascii=False), flush=True)
        score()
    else:
        {"prepare": prepare, "run": run, "score": score}[sys.argv[1]]()
