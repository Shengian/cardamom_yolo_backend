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

print(json.dumps(response.json, ensure_ascii=False, indent=2))

