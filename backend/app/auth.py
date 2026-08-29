import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

SECRET = os.environ.get("AEGIS_SECRET", "aegis-fyp-demo-secret")

HOSPITALS = [
    {
        "id": 0,
        "slug": "st-helens",
        "name": "St. Helen's Infirmary",
        "city": "Colombo",
        "focus": "Adult endocrinology",
    },
    {
        "id": 1,
        "slug": "riverside",
        "name": "Riverside General",
        "city": "Kandy",
        "focus": "Acute medical wards",
    },
    {
        "id": 2,
        "slug": "oakridge",
        "name": "Oakridge Medical Centre",
        "city": "Galle",
        "focus": "Regional diabetes clinic",
    },
]

USERS: dict[str, dict[str, Any]] = {
    "admin": {
        "password": "aegis-admin",
        "role": "admin",
        "hospital_id": None,
        "name": "Network coordinator",
    },
    "sthelens": {
        "password": "ward-demo",
        "role": "hospital",
        "hospital_id": 0,
        "name": "St. Helen's Infirmary",
    },
    "riverside": {
        "password": "ward-demo",
        "role": "hospital",
        "hospital_id": 1,
        "name": "Riverside General",
    },
    "oakridge": {
        "password": "ward-demo",
        "role": "hospital",
        "hospital_id": 2,
        "name": "Oakridge Medical Centre",
    },
}


def _sign(payload: str) -> str:
    return hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(username: str) -> str:
    body = json.dumps({"u": username, "exp": time.time() + 60 * 60 * 12}, separators=(",", ":"))
    packed = base64.urlsafe_b64encode(body.encode()).decode()
    return f"{packed}.{_sign(body)}"


def parse_token(token: str) -> dict[str, Any] | None:
    try:
        packed, sig = token.rsplit(".", 1)
        body = base64.urlsafe_b64decode(packed.encode()).decode()
        if not hmac.compare_digest(sig, _sign(body)):
            return None
        data = json.loads(body)
        if data["exp"] < time.time():
            return None
        user = USERS.get(data["u"])
        if not user:
            return None
        return {"username": data["u"], **{k: v for k, v in user.items() if k != "password"}}
    except (ValueError, json.JSONDecodeError, KeyError):
        return None


def hospital_by_id(hospital_id: int) -> dict[str, Any]:
    return next(h for h in HOSPITALS if h["id"] == hospital_id)
