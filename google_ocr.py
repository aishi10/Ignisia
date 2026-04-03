import json
import pandas as pd
from pathlib import Path
import os
import zipfile
import argparse
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict

# Google Vision
from google.cloud import vision

# ---------------- CONFIG ----------------
CONF_THRESH = 55

# Initialize client ONCE (important)
client = vision.ImageAnnotatorClient.from_service_account_json(
    "/Users/chaku/Downloads/ocrmodelignisia-95833f16e186.json"
)

# ---------------- HELPERS ----------------

def clean_text(text):
    import re
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_filenames(names):
    import re
    patterns = [
        r"^(\w+)[_\-]sheet(\d+)\.(png|jpg|jpeg)$",
        r".*?[_\-\s]?(\w+)[_\-\s]p(\d+)\.(png|jpg|jpeg)$",
        r".*?[_\-\s](\w+)[_\-\s]page(\d+)\.(png|jpg|jpeg)$",
        r"^(\w+)[_\-](\d+)\.(png|jpg|jpeg)$",
    ]

    grouped = defaultdict(dict)

    for name in names:
        base = Path(name).name
        if not base.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        for pat in patterns:
            m = re.match(pat, base, re.IGNORECASE)
            if m:
                grouped[m.group(1)][int(m.group(2))] = name
                break

    return dict(grouped)


# ---------------- GOOGLE OCR ----------------

def google_ocr_bytes(img_bytes):
    image = vision.Image(content=img_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        print("Error:", response.error.message)
        return {"text": "", "confidence": 0}

    text = response.full_text_annotation.text

    # Google doesn't give direct confidence → assume high
    return {"text": clean_text(text), "confidence": 90}


def process_page(pil_img):
    import io

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")

    result = google_ocr_bytes(buffer.getvalue())

    return result


# ---------------- MAIN PIPELINE ----------------

def run_pipeline(source):
    print("GOOGLE OCR PIPELINE\n")

    if os.path.isdir(source):
        all_names = [
            str(p) for p in Path(source).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        read_file = lambda x: open(x, "rb").read()
    else:
        zf = zipfile.ZipFile(source)
        all_names = zf.namelist()
        read_file = lambda x: zf.read(x)

    grouped = parse_filenames(all_names)

    page_jobs = []
    for sid, pages in grouped.items():
        for page_num, page_path in sorted(pages.items()):
            page_jobs.append((sid, page_num, page_path))

    page_outputs = defaultdict(dict)

    for sid, page_num, page_path in tqdm(page_jobs, desc="OCR pages"):
        img = Image.open(BytesIO(read_file(page_path)))
        page_outputs[sid][page_num] = process_page(img)

    results = []
    flagged = 0

    for sid in sorted(page_outputs):
        ordered = page_outputs[sid]

        full_text = " ".join(
            ordered[p]["text"] for p in sorted(ordered)
        )

        confs = [
            ordered[p]["confidence"]
            for p in sorted(ordered)
            if ordered[p]["confidence"] > 0
        ]

        avg_conf = sum(confs) / len(confs) if confs else 0

        row = {
            "student_id": str(sid),
            "full_text": clean_text(full_text),
            "avg_confidence": float(round(avg_conf, 1)),
            "flagged": bool(avg_conf < CONF_THRESH)
        }

        results.append(row)
        if row["flagged"]:
            flagged += 1

    avg = sum(r["avg_confidence"] for r in results) / len(results) if results else 0

    print("\n" + "-" * 40)
    print(f"Avg confidence: {round(avg, 1)}")
    print(f"Flagged: {flagged}")
    print("-" * 40)

    output_dir = Path("ocr_output")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)

    print(f"Saved JSON: {output_dir / 'results.json'}")
    print(f"Saved CSV: {output_dir / 'results.csv'}")

    return results


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--zip", type=str, default=None)
    args = parser.parse_args()

    if args.folder:
        run_pipeline(args.folder)
    elif args.zip:
        run_pipeline(args.zip)
    else:
        print("Provide --folder or --zip")
