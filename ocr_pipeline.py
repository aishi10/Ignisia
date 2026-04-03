"""
ocr_pipeline_v2.py
Improved OCR pipeline (better accuracy + confidence)
"""

import os
import cv2
import numpy as np
import pytesseract
import zipfile
import json
import time
import argparse
import re
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict


# --- CONFIG ---
PSM         = 4          # improved default
CONF_THRESH = 40         # lower threshold for handwritten text
OUTPUT_DIR  = Path("ocr_output")


# --- FILENAME PARSING ---
def parse_filenames(names):
    patterns = [
        r"^(\w+)[_\-]sheet(\d+)\.png",
        r".*?[_\-\s]?(\w+)[_\-\s]p(\d+)\.png",
        r".*?[_\-\s](\w+)[_\-\s]page(\d+)\.png",
        r"^(\w+)[_\-](\d)\.png",
        r".*?(\d+)pg(\d+)\.png",
        r"^(\d+)-(\d+)\.png",
    ]
    grouped = defaultdict(dict)

    for name in names:
        if not name.lower().endswith(".png"):
            continue

        basename = Path(name).name

        for pat in patterns:
            m = re.match(pat, basename, re.IGNORECASE)
            if m:
                student_id = m.group(1)
                page_num   = int(m.group(2))
                grouped[student_id][page_num] = name
                break

    return dict(grouped)


# --- PREPROCESS (LIGHT + SAFE) ---
def preprocess(pil_img):
    img  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # light blur (avoid destroying text)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive threshold (better for uneven lighting)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return binary


# --- OCR ---
def ocr_image(img, psm):
    pil = Image.fromarray(img)
    config = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(
        pil,
        lang="eng",
        config=config,
        output_type=pytesseract.Output.DICT
    )

    words, confs = [], []

    for word, conf in zip(data["text"], data["conf"]):
        conf = int(conf)
        if word.strip() and conf > 30:   # filter weak noise
            words.append(word)
            confs.append(conf)

    return {
        "text": " ".join(words),
        "confidence": round(sum(confs) / len(confs), 1) if confs else 0.0,
        "word_count": len(words),
    }


# --- MAIN PIPELINE ---
def run_pipeline(source, psm):
    OUTPUT_DIR.mkdir(exist_ok=True)
    source_path = Path(source)

    # detect zip or folder
    if zipfile.is_zipfile(source):
        print("Reading from ZIP...\n")
        zf = zipfile.ZipFile(source, "r")
        all_names = zf.namelist()

        def read_file(name):
            return zf.read(name)

    elif os.path.isdir(source):
        print("Reading from folder...\n")
        all_names = [str(p) for p in source_path.rglob("*.png")]

        def read_file(name):
            with open(name, "rb") as f:
                return f.read()
    else:
        raise ValueError("Source must be a valid folder or zip file")

    print(f"PNGs     : {len(all_names)}")

    grouped = parse_filenames(all_names)
    print(f"Students : {len(grouped)} detected")
    print(f"PSM mode : {psm}\n")

    if not grouped:
        print("❌ No valid students detected.\n")
        return

    results = []
    flagged = 0
    total_ms = 0

    for student_id, pages in tqdm(grouped.items(), desc="Processing"):
        try:
            t0 = time.time()
            page_results = {}

            for page_num in sorted(pages.keys()):
                img_bytes = read_file(pages[page_num])
                pil_img   = Image.open(BytesIO(img_bytes))

                clean = preprocess(pil_img)
                page_results[page_num] = ocr_image(clean, psm)

            full_text = " ".join(
                page_results[p]["text"] for p in sorted(page_results)
            ).strip()

            all_confs = [
                page_results[p]["confidence"]
                for p in page_results if page_results[p]["confidence"] > 0
            ]

            avg_conf = round(sum(all_confs) / len(all_confs), 1) if all_confs else 0.0
            total_words = sum(page_results[p]["word_count"] for p in page_results)

            elapsed_ms = round((time.time() - t0) * 1000)
            total_ms += elapsed_ms

            is_flagged = avg_conf < CONF_THRESH
            if is_flagged:
                flagged += 1

            record = {
                "student_id": student_id,
                "full_text": full_text,
                "page_texts": {
                    str(p): page_results[p]["text"] for p in sorted(page_results)
                },
                "avg_confidence": avg_conf,
                "total_word_count": total_words,
                "pages_found": sorted(pages.keys()),
                "flagged": is_flagged,
                "processing_ms": elapsed_ms,
            }

            results.append(record)

            (OUTPUT_DIR / f"student_{student_id}.json").write_text(
                json.dumps(record, indent=2)
            )

        except Exception as e:
            print(f"\nERROR processing {student_id}: {e}")

    # --- FINAL ---
    n = len(results)

    cost_log = {
        "backend": "tesseract",
        "source": source,
        "total_students": n,
        "total_pages_processed": n * 2,
        "flagged_students": flagged,
        "flag_rate_pct": round(flagged / n * 100, 1) if n else 0,
        "total_time_s": round(total_ms / 1000, 2),
        "avg_time_per_student_ms": round(total_ms / n, 1) if n else 0,
        "api_cost_usd": 0.0,
    }

    (OUTPUT_DIR / "all_answers.json").write_text(
        json.dumps({"cost_log": cost_log, "answers": results}, indent=2)
    )

    avg_conf_all = round(
        sum(r["avg_confidence"] for r in results) / n, 1
    ) if n else 0

    print("\n" + "─"*50)
    print(f"Done. {n} students processed")
    print(f"Avg confidence : {avg_conf_all}")
    print(f"Flagged        : {flagged}")
    print(f"Output folder  : {OUTPUT_DIR}/")
    print("─"*50 + "\n")


# --- INSPECT ---
def inspect_output():
    path = OUTPUT_DIR / "all_answers.json"
    if not path.exists():
        print("No output yet.")
        return
    data = json.loads(path.read_text())
    print(json.dumps(data["cost_log"], indent=2))


# --- ENTRY ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--zip")
    parser.add_argument("--psm", default=PSM, type=int)
    parser.add_argument("--inspect", action="store_true")

    args = parser.parse_args()

    if args.inspect:
        inspect_output()
    elif args.folder:
        run_pipeline(args.folder, args.psm)
    elif args.zip:
        run_pipeline(args.zip, args.psm)
    else:
        parser.print_help()
