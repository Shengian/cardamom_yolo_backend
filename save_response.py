from __future__ import annotations

import json

from app import app

client = app.test_client()

with open("elachi_test.jpg", "rb") as f:
    response = client.post(
        "/predict",
        data={"image": (f, "elachi_test.jpg")},
        content_type="multipart/form-data",
    )

payload = json.dumps(response.json, ensure_ascii=False, indent=2)

with open("api_predict.json", "w", encoding="utf-8") as f:
    f.write(payload)


