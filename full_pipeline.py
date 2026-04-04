import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

from cost_efficiency_logger import (
    DEFAULT_INPUT_COST_PER_1M,
    DEFAULT_OUTPUT_COST_PER_1M,
    generate_cost_efficiency_summary,
)
from embedding import cluster_answers
from run_pipeline import run_grading_pipeline


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--zip", type=str, default=None)
    parser.add_argument(
        "--answer-key",
        type=str,
        default=None,
        help="Optional path to the answer-key CSV used for grading.",
    )
    parser.add_argument(
        "--group-by",
        choices=["auto", "folder", "filename", "single"],
        default="auto",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSON/CSV manifest mapping files to document IDs and page numbers.",
    )
    parser.add_argument("--model-name", type=str, default="gpt-4.1-mini-estimate")
    parser.add_argument("--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M)
    parser.add_argument("--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M)
    return parser


def print_cost_summary(summary, json_log_path, csv_log_path):
    print("\n=== COST & EFFICIENCY SUMMARY ===")
    print(f"Sheets processed: {summary['sheet_count']}")
    print(
        "Stage times (s): "
        f"OCR={summary['stage_times']['ocr_seconds']}, "
        f"Clustering={summary['stage_times']['clustering_seconds']}, "
        f"Grading={summary['stage_times']['grading_seconds']}, "
        f"Total={summary['stage_times']['total_seconds']}"
    )

    for batch in summary["batches"]:
        print(
            f"Batch {batch['batch_id']} ({batch['sheet_range']}, {batch['sheet_count']} sheets): "
            f"tokens={batch['estimated_total_tokens']} "
            f"(in={batch['estimated_input_tokens']}, out={batch['estimated_output_tokens']}), "
            f"time={batch['estimated_processing_seconds']}s, "
            f"cost=${batch['estimated_cost_usd']}"
        )

    print(summary["token_estimation_note"])
    print(f"Cost log saved to: {json_log_path}")
    print(f"Efficiency CSV saved to: {csv_log_path}")


def build_run_dir(base_dir, source):
    stem = Path(str(source)).stem or "run"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem).strip("_") or "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "runs" / f"{timestamp}_{safe_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_full_pipeline(
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
    run_dir = build_run_dir(backend_dir, source)
    stage_times = {}

    print("\n=== STAGE 1: OCR ===")
    stage_start = time.perf_counter()
    run_ocr_pipeline(
        source,
        group_by=group_by,
        manifest_path=manifest_path,
        output_dir=run_dir / "ocr_output",
    )
    stage_times["ocr_seconds"] = round(time.perf_counter() - stage_start, 2)

    print("\n=== STAGE 2: CLUSTERING ===")
    results_json_path = run_dir / "ocr_output" / "results.json"
    clustered_csv_path = run_dir / "clustered_answers.csv"
    clustered_json_path = run_dir / "clustered_answers.json"
    stage_start = time.perf_counter()
    cluster_answers(
        results_json_path=results_json_path,
        output_csv_path=clustered_csv_path,
        output_json_path=clustered_json_path,
        answer_key_path=answer_key_path,
    )
    stage_times["clustering_seconds"] = round(time.perf_counter() - stage_start, 2)

    print("\n=== STAGE 3: GRADING ===")
    output_path = run_dir / "grading_output.json"
    stage_start = time.perf_counter()
    final_output = run_grading_pipeline(
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
        log_dir=run_dir / "logs",
    )
    print_cost_summary(summary, json_log_path, csv_log_path)

    return {
        "results": final_output,
        "run_dir": run_dir,
        "output_path": output_path,
        "clustered_csv_path": clustered_csv_path,
        "results_json_path": results_json_path,
        "summary": summary,
    }


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    source = args.folder or args.zip
    if not source:
        raise SystemExit("Provide --folder or --zip")

    run_full_pipeline(
        source,
        answer_key_path=args.answer_key,
        group_by=args.group_by,
        manifest_path=args.manifest,
        model_name=args.model_name,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
