from __future__ import annotations

import os
import tempfile

from flask import Flask, jsonify, request

from predict import run_prediction

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    return {"status": "ok"}, 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file in form-data."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        result = run_prediction(
            temp_path,
            save_images=False,
            write_text=False,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return jsonify(result)


if __name__ == "__main__":
    # For local testing; Render will use gunicorn app:app
    app.run(host="0.0.0.0", port=8000, debug=False)



