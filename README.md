# Backend Pipeline

This backend contains the connected OCR, clustering, grading, and reporting flow for Ignisia.

## Main Entry Points

- `full_pipeline.py`
  Runs OCR -> clustering -> grading and prints the cost/efficiency summary at the end.
- `cost_efficiency_logger.py`
  Runs the same flow and writes batch-level cost and timing logs.
- `embedding.py`
  Reads OCR JSON output and creates clustered CSV/JSON files.
- `run_pipeline.py`
  Grades clustered answers against the answer key.

## Typical Run

From this folder:

```bash
python3 full_pipeline.py --folder /Users/chaku/Desktop/Ignisia-main/dataset/english
```

If image naming is irregular, you can force grouping mode:

```bash
python3 full_pipeline.py --folder /Users/chaku/Desktop/Ignisia-main/dataset/english --group-by single
```

## Inputs

- OCR source folder or zip of answer-sheet images
- `Answer_Key_Q1_Q2.csv`

## Generated Outputs

- `ocr_output/results.json`
- `ocr_output/results.csv`
- `final_clustered_grades.csv`
- `final_clustered_grades.json`
- `output.json`
- `logs/cost_efficiency_log.json`
- `logs/cost_efficiency_log.csv`

## Notes

- Cluster suggestions are produced question-wise for `Q1` and `Q2`.
- Suggested marks are shown in `suggested_marks/total_marks` format.
- Cost and token usage are estimated, not direct LLM API billing data.
