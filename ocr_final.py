import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google.cloud import vision
from tqdm import tqdm

load_dotenv()

CONF_THRESH = 55
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def get_client():
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Please add it to your .env file or export it in your shell."
        )
    return vision.ImageAnnotatorClient.from_service_account_json(cred_path)


def clean_text(text):
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def natural_sort_key(value):
    text = str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def detect_page_number(path_like):
    filename = Path(path_like).name
    patterns = [
        r"(?:page|pg|p|sheet)[_\-\s]*(\d+)",
        r"[_\-\s](\d+)(?:\.[a-z]+)?$",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def normalize_document_id(value, fallback_index):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_")
    if cleaned:
        return cleaned
    return f"doc_{fallback_index:03d}"


def parse_filename_groups(names):
    patterns = [
        r"^(\w+)[_\-]sheet(\d+)\.(png|jpg|jpeg)$",
        r".*?[_\-\s]?(\w+)[_\-\s]p(\d+)\.(png|jpg|jpeg)$",
        r".*?[_\-\s](\w+)[_\-\s]page(\d+)\.(png|jpg|jpeg)$",
        r"^(\w+)[_\-](\d+)\.(png|jpg|jpeg)$",
    ]

    grouped = defaultdict(list)
    for name in names:
        base = Path(name).name
        if Path(base).suffix.lower() not in IMAGE_SUFFIXES:
            continue

        matched = False
        for pattern in patterns:
            match = re.match(pattern, base, re.IGNORECASE)
            if match:
                grouped[match.group(1)].append({
                    "path": name,
                    "page_num": int(match.group(2)),
                })
                matched = True
                break

        if not matched:
            return None

    return grouped if grouped else None


def parse_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if manifest_path.suffix.lower() == ".json":
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif manifest_path.suffix.lower() == ".csv":
        rows = pd.read_csv(manifest_path).to_dict(orient="records")
    else:
        raise ValueError("Manifest must be .json or .csv")

    grouped = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        file_path = row.get("file") or row.get("path")
        if not file_path:
            continue

        document_id = normalize_document_id(
            row.get("document_id") or row.get("student_id"),
            index,
        )
        page_num = row.get("page")
        grouped[document_id].append({
            "path": str(file_path),
            "page_num": int(page_num) if str(page_num).strip() else None,
        })

    return grouped


def discover_images(source):
    if os.path.isdir(source):
        base_dir = Path(source)
        image_names = [
            str(path.relative_to(base_dir))
            for path in base_dir.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]

        def read_file(name):
            return (base_dir / name).read_bytes()

        return image_names, read_file

    if os.path.isfile(source) and source.lower().endswith(".zip"):
        archive = zipfile.ZipFile(source)
        image_names = [
            name for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        ]

        def read_file(name):
            return archive.read(name)

        return image_names, read_file

    raise FileNotFoundError(f"Input path not found or unsupported: {source}")


def group_pages(image_names, mode="auto", manifest_path=None):
    if manifest_path:
        grouped = parse_manifest(manifest_path)
        strategy = "manifest"
    elif mode == "filename":
        grouped = parse_filename_groups(image_names)
        if grouped is None:
            raise ValueError("Filename grouping failed for this dataset.")
        strategy = "filename"
    elif mode == "folder":
        grouped = defaultdict(list)
        for index, name in enumerate(sorted(image_names, key=natural_sort_key), start=1):
            parent = Path(name).parent
            document_id = normalize_document_id(
                parent.name if str(parent) not in {".", ""} else None,
                index,
            )
            grouped[document_id].append({
                "path": name,
                "page_num": detect_page_number(name),
            })
        strategy = "folder"
    elif mode == "single":
        grouped = defaultdict(list)
        for index, name in enumerate(sorted(image_names, key=natural_sort_key), start=1):
            grouped[normalize_document_id(Path(name).stem, index)].append({
                "path": name,
                "page_num": detect_page_number(name),
            })
        strategy = "single"
    else:
        grouped = parse_filename_groups(image_names)
        if grouped:
            strategy = "filename"
        else:
            folder_has_structure = any(Path(name).parent != Path(".") for name in image_names)
            grouped = defaultdict(list)
            for index, name in enumerate(sorted(image_names, key=natural_sort_key), start=1):
                if folder_has_structure:
                    parent = Path(name).parent
                    document_id = normalize_document_id(parent.name, index)
                    strategy = "folder"
                else:
                    document_id = normalize_document_id(Path(name).stem, index)
                    strategy = "single"

                grouped[document_id].append({
                    "path": name,
                    "page_num": detect_page_number(name),
                })

    normalized = {}
    for index, document_id in enumerate(sorted(grouped, key=natural_sort_key), start=1):
        normalized_id = normalize_document_id(document_id, index)
        ordered_pages = sorted(
            grouped[document_id],
            key=lambda item: (
                item["page_num"] is None,
                item["page_num"] if item["page_num"] is not None else 10**9,
                natural_sort_key(item["path"]),
            ),
        )
        normalized[normalized_id] = ordered_pages

    return normalized, strategy


def google_ocr_bytes(img_bytes, client):
    image = vision.Image(content=img_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        print("Error:", response.error.message)
        return {"text": "", "confidence": 0}

    text = response.full_text_annotation.text
    return {"text": clean_text(text), "confidence": 90}


def process_page(pil_img, client):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return google_ocr_bytes(buffer.getvalue(), client)


def run_pipeline(source, group_by="auto", manifest_path=None):
    print("GOOGLE OCR PIPELINE\n")

    client = get_client()
    image_names, read_file = discover_images(source)
    grouped, grouping_strategy = group_pages(
        image_names,
        mode=group_by,
        manifest_path=manifest_path,
    )

    if not grouped:
        raise ValueError("No images found to process.")

    page_jobs = []
    for document_id, pages in grouped.items():
        for sequence_index, page in enumerate(pages, start=1):
            page_jobs.append((
                document_id,
                page["page_num"] or sequence_index,
                page["path"],
            ))

    page_outputs = defaultdict(dict)

    for document_id, page_num, page_path in tqdm(page_jobs, desc="OCR pages"):
        img = Image.open(BytesIO(read_file(page_path)))
        page_outputs[document_id][page_num] = {
            **process_page(img, client),
            "source_path": page_path,
        }

    results = []
    flagged = 0

    for index, document_id in enumerate(sorted(page_outputs, key=natural_sort_key), start=1):
        ordered = page_outputs[document_id]
        ordered_pages = [ordered[p] for p in sorted(ordered)]

        full_text = " ".join(page["text"] for page in ordered_pages)
        confidences = [page["confidence"] for page in ordered_pages if page["confidence"] > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        normalized_id = normalize_document_id(document_id, index)
        row = {
            "student_id": normalized_id,
            "document_id": normalized_id,
            "source_files": [page["source_path"] for page in ordered_pages],
            "grouping_strategy": grouping_strategy,
            "full_text": str(clean_text(full_text)),
            "avg_confidence": float(round(avg_conf, 1)),
            "flagged": bool(avg_conf < CONF_THRESH),
        }

        results.append(row)
        if row["flagged"]:
            flagged += 1

    avg = sum(r["avg_confidence"] for r in results) / len(results) if results else 0

    print("\n" + "-" * 40)
    print(f"Grouping strategy: {grouping_strategy}")
    print(f"Documents processed: {len(results)}")
    print(f"Avg confidence: {round(avg, 1)}")
    print(f"Flagged: {flagged}")
    print("-" * 40)

    output_dir = Path("ocr_output")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)

    print(f"Saved JSON: {output_dir / 'results.json'}")
    print(f"Saved CSV: {output_dir / 'results.csv'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--zip", type=str, default=None)
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
    args = parser.parse_args()

    source = args.folder or args.zip
    if not source:
        print("Provide --folder or --zip")
    else:
        run_pipeline(source, group_by=args.group_by, manifest_path=args.manifest)
