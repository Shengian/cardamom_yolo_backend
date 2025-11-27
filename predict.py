from __future__ import annotations

import json
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO

from solutions_data import SOLUTIONS

MODEL_PATH = Path("runs/detect/train3/weights/best.pt")
DEFAULT_IMAGE = "elachi_test.jpg"
CLASS_NAMES = {
    0: "Cardamom_HL",
    1: "C_Blight",
    2: "Healthy",
}


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    print("Loading Model:", MODEL_PATH)
    return YOLO(str(MODEL_PATH))


def build_solution_text(label: str, data: Dict[str, Any]) -> str:
    text = f"\n=== {label} ===\n\n"

    text += "English:\n"
    text += "- " + data["english"]["problem"] + "\n"
    for sentence in data["english"]["solutions"]:
        text += "- " + sentence + "\n"

    text += "\nTamil:\n"
    text += "- " + data["tamil"]["problem"] + "\n"
    for sentence in data["tamil"]["solutions"]:
        text += "- " + sentence + "\n"

    return text


def run_prediction(
    image_path: str,
    *,
    save_images: bool = True,
    write_text: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    if verbose:
        print("\nRunning prediction...")
    model = get_model()
    results = model.predict(source=image_path, save=save_images)

    if len(results[0].boxes) == 0:
        detected = "Healthy"
    else:
        cls_id = int(results[0].boxes[0].cls)
        detected = CLASS_NAMES[cls_id]

    if verbose:
        print("\nDetected:", detected)

    data = SOLUTIONS[detected]

    if write_text:
        text = build_solution_text(detected, data)
        with open("output_solution.txt", "w", encoding="utf-8") as f:
            f.write(text)
        if verbose:
            print("\n✔ output_solution.txt CREATED successfully!")

    result_payload = {
        "detected": detected,
        "details": data,
    }
    if verbose:
        print("\nJSON payload (for cloud workflows):")
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))
    return result_payload


def main() -> None:
    print("\n🔥🔥🔥 RUNNING predict.py 🔥🔥🔥")

    image_path = DEFAULT_IMAGE
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    print("Using Image:", image_path)

    try:
        run_prediction(image_path)
    except Exception as exc:
        print("\n❌ ERROR OCCURRED")
        print(exc)
        traceback.print_exc()


if __name__ == "__main__":
    main()
