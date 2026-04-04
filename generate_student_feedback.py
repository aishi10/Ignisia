import argparse
from pathlib import Path

from feedback_generator import generate_feedback_packages
from full_pipeline import run_full_pipeline


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--zip", type=str, default=None)
    parser.add_argument("--reviews", type=str, default=None)
    parser.add_argument("--clustered-csv", type=str, default=None)
    parser.add_argument("--answer-key", type=str, default=None)
    parser.add_argument("--grading-output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--group-by",
        choices=["auto", "folder", "filename", "single"],
        default="auto",
    )
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument(
        "--skip-full-pipeline",
        action="store_true",
        help="Use existing clustered/grading outputs instead of rerunning OCR, clustering, and grading.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent
    clustered_csv_path = args.clustered_csv or backend_dir / "final_clustered_grades.csv"
    grading_output_path = args.grading_output or backend_dir / "output.json"
    output_dir = args.output_dir or backend_dir / "exports"

    if not args.skip_full_pipeline:
        source = args.folder or args.zip
        if not source:
            raise SystemExit("Provide --folder or --zip, or pass --skip-full-pipeline.")

        print("Running full pipeline before feedback generation...")
        result_bundle = run_full_pipeline(
            source=source,
            answer_key_path=args.answer_key or backend_dir / "Answer_Key_Q1_Q2.csv",
            group_by=args.group_by,
            manifest_path=args.manifest,
        )
        clustered_csv_path = result_bundle["clustered_csv_path"]
        grading_output_path = result_bundle["output_path"]
        output_dir = result_bundle["run_dir"] / "exports"

    print("Starting student feedback generation...")
    result = generate_feedback_packages(
        review_path=args.reviews,
        clustered_csv_path=clustered_csv_path,
        answer_key_path=args.answer_key or backend_dir / "Answer_Key_Q1_Q2.csv",
        grading_output_path=grading_output_path,
        output_dir=output_dir,
    )

    print(f"Review data used from: {result['review_path']}")
    print(f"Student feedback saved to: {result['feedback_path']}")
    print(f"PDF exports saved to: {result['pdf_dir']}")
    print(f"Email exports saved to: {result['email_dir']}")
