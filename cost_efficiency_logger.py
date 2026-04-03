import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from embedding import cluster_answers
from run_pipeline import run_grading_pipeline


DEFAULT_INPUT_COST_PER_1M = 0.15
DEFAULT_OUTPUT_COST_PER_1M = 0.60
SHEETS_PER_BATCH = 20


def estimate_tokens(text):
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 4))


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--zip", type=str, default=None)
    parser.add_argument("--answer-key", type=str, default=None)
    parser.add_argument(
        "--group-by",
        choices=["auto", "folder", "filename", "single"],
        default="auto",
    )
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini-estimate")
    parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
    parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
    return parser


def generate_cost_efficiency_summary(
    source,
    answer_key_path,
    group_by,
    manifest_path,
    model_name,
    input_cost_per_1m,
    output_cost_per_1m,
    stage_times,
    results_json_path,
    output_path,
):
    backend_dir = Path(__file__).resolve().parent
    log_dir = backend_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    answer_key_path = Path(answer_key_path)
    ocr_rows = load_json(results_json_path)
    grading_output = load_json(output_path)
    sheet_count = len(ocr_rows)

    with open(answer_key_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        answer_key_rows = list(reader)

    answer_key_prompt_text = " ".join(
        " ".join([
            row.get("Question", ""),
            row.get("Model_Answer", ""),
            row.get("Required_Elements", ""),
        ])
        for row in answer_key_rows
    )
    answer_key_tokens = estimate_tokens(answer_key_prompt_text)

    per_sheet_logs = []
    grading_reasons = []
    for _, clusters in grading_output.items():
        for cluster in clusters:
            semantic = cluster.get("semantic_evaluation", {})
            grading_reasons.append(semantic.get("reason", ""))

    estimated_output_tokens_total = sum(estimate_tokens(reason) for reason in grading_reasons)

    for row in ocr_rows:
        text_tokens = estimate_tokens(row.get("full_text", ""))
        per_sheet_logs.append({
            "student_id": row.get("student_id", ""),
            "document_id": row.get("document_id", row.get("student_id", "")),
            "estimated_input_tokens": text_tokens,
        })

    batches = []
    for batch_index, start in enumerate(range(0, sheet_count, SHEETS_PER_BATCH), start=1):
        batch_rows = per_sheet_logs[start:start + SHEETS_PER_BATCH]
        batch_size = len(batch_rows)
        batch_input_tokens = sum(item["estimated_input_tokens"] for item in batch_rows)
        proportional_answer_key_tokens = math.ceil(answer_key_tokens * (batch_size / max(1, sheet_count)))
        proportional_output_tokens = math.ceil(estimated_output_tokens_total * (batch_size / max(1, sheet_count)))
        total_input_tokens = batch_input_tokens + proportional_answer_key_tokens
        total_output_tokens = proportional_output_tokens
        total_tokens = total_input_tokens + total_output_tokens

        time_share = batch_size / max(1, sheet_count)
        estimated_processing_seconds = round(stage_times["total_seconds"] * time_share, 2)
        estimated_ocr_seconds = round(stage_times["ocr_seconds"] * time_share, 2)
        estimated_clustering_seconds = round(stage_times["clustering_seconds"] * time_share, 2)
        estimated_grading_seconds = round(stage_times["grading_seconds"] * time_share, 2)

        estimated_cost_usd = round(
            (total_input_tokens / 1_000_000) * input_cost_per_1m +
            (total_output_tokens / 1_000_000) * output_cost_per_1m,
            6,
        )

        batches.append({
            "batch_id": batch_index,
            "sheet_range": f"{start + 1}-{start + batch_size}",
            "sheet_count": batch_size,
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_total_tokens": total_tokens,
            "estimated_processing_seconds": estimated_processing_seconds,
            "estimated_ocr_seconds": estimated_ocr_seconds,
            "estimated_clustering_seconds": estimated_clustering_seconds,
            "estimated_grading_seconds": estimated_grading_seconds,
            "estimated_cost_usd": estimated_cost_usd,
            "model_name": model_name,
        })

    summary = {
        "source": str(source),
        "answer_key_path": str(answer_key_path),
        "group_by": group_by,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "model_name": model_name,
        "token_estimation_note": (
            "Estimated token usage only. The current backend does not call an LLM directly; "
            "tokens are approximated from OCR text, answer-key context, and generated grading reasons."
        ),
        "sheets_per_batch": SHEETS_PER_BATCH,
        "sheet_count": sheet_count,
        "stage_times": stage_times,
        "batches": batches,
    }

    json_log_path = log_dir / "cost_efficiency_log.json"
    csv_log_path = log_dir / "cost_efficiency_log.csv"

    with open(json_log_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(csv_log_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "batch_id",
                "sheet_range",
                "sheet_count",
                "estimated_input_tokens",
                "estimated_output_tokens",
                "estimated_total_tokens",
                "estimated_processing_seconds",
                "estimated_ocr_seconds",
                "estimated_clustering_seconds",
                "estimated_grading_seconds",
                "estimated_cost_usd",
                "model_name",
            ],
        )
        writer.writeheader()
        writer.writerows(batches)

    return summary, json_log_path, csv_log_path


def run_pipeline_with_logging(
    source,
    answer_key_path=None,
    group_by="auto",
    manifest_path=None,
    model_name="gpt-4.1-mini-estimate",
    input_cost_per_1m=DEFAULT_INPUT_COST_PER_1M,
    output_cost_per_1m=DEFAULT_OUTPUT_COST_PER_1M,
):
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    sys.path.insert(0, str(project_root))

    from ocr_final import run_pipeline as run_ocr_pipeline

    answer_key_path = Path(answer_key_path or backend_dir / "Answer_Key_Q1_Q2.csv")
    stage_times = {}

    stage_start = time.perf_counter()
    run_ocr_pipeline(source, group_by=group_by, manifest_path=manifest_path)
    stage_times["ocr_seconds"] = round(time.perf_counter() - stage_start, 2)

    backend_ocr_results = backend_dir / "ocr_output" / "results.json"
    project_ocr_results = project_root / "ocr_output" / "results.json"
    results_json_path = backend_ocr_results if backend_ocr_results.exists() else project_ocr_results

    clustered_csv_path = backend_dir / "final_clustered_grades.csv"
    clustered_json_path = backend_dir / "final_clustered_grades.json"

    stage_start = time.perf_counter()
    cluster_answers(
        results_json_path=results_json_path,
        output_csv_path=clustered_csv_path,
        output_json_path=clustered_json_path,
    )
    stage_times["clustering_seconds"] = round(time.perf_counter() - stage_start, 2)

    output_path = backend_dir / "output.json"
    stage_start = time.perf_counter()
    run_grading_pipeline(
        clustered_csv_path=clustered_csv_path,
        answer_key_path=answer_key_path,
        output_path=output_path,
    )
    stage_times["grading_seconds"] = round(time.perf_counter() - stage_start, 2)
    stage_times["total_seconds"] = round(sum(stage_times.values()), 2)

    summary, json_log_path, csv_log_path = generate_cost_efficiency_summary(
        source=source,
        answer_key_path=answer_key_path,
        group_by=group_by,
        manifest_path=manifest_path,
        model_name=model_name,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        stage_times=stage_times,
        results_json_path=results_json_path,
        output_path=output_path,
    )

    print(f"\nCost log saved to: {json_log_path}")
    print(f"Efficiency CSV saved to: {csv_log_path}")
    return summary


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    source = args.folder or args.zip
    if not source:
        raise SystemExit("Provide --folder or --zip")

    run_pipeline_with_logging(
        source=source,
        answer_key_path=args.answer_key,
        group_by=args.group_by,
        manifest_path=args.manifest,
        model_name=args.model_name,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
