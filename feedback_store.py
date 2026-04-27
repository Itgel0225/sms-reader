"""Хэрэглэгчийн засалуудыг feedback.json файлд хадгалж GitHub-т автоматаар commit хийнэ.

Render-ийн filesystem ephemeral учир restart хийгдэхэд алдагдахгүй байхын тулд
GitHub repo-д шууд push хийдэг. Дараагийн deploy дээр feedback.json нь шинэлэг
байж, бүх засал ML model-д сурч ирнэ.
"""

import os
import json
import base64
from threading import Lock
from typing import List, Tuple

import requests

GITHUB_REPO = os.getenv("GITHUB_REPO", "Itgel0225/sms-reader")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
FEEDBACK_FILE = "feedback.json"

_lock = Lock()


def load_feedback() -> List[Tuple[str, str]]:
    """Локал файлаас засалуудыг унших → [(sms_text, category), ...]."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [tuple(item) for item in data if isinstance(item, list) and len(item) == 2]
    except Exception as e:
        print(f"⚠️ feedback.json уншихад алдаа: {e}")
        return []


def _save_local(samples: List[Tuple[str, str]]) -> None:
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump([list(s) for s in samples], f, ensure_ascii=False, indent=2)


def _push_to_github(samples: List[Tuple[str, str]]) -> bool:
    """feedback.json-г GitHub-т commit хийнэ."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("⚠️ GITHUB_TOKEN env var тохируулаагүй — GitHub-т push хийгдэхгүй")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FEEDBACK_FILE}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Одоогийн файлын SHA-г авах (update хийхэд хэрэгтэй)
    sha = None
    try:
        r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=10)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception as e:
        print(f"⚠️ GitHub SHA авч чадсангүй: {e}")

    content = json.dumps([list(s) for s in samples], ensure_ascii=False, indent=2)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    body = {
        "message": f"Feedback: {len(samples)} нийт жишээ",
        "content": content_b64,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        body["sha"] = sha

    try:
        r = requests.put(url, headers=headers, json=body, timeout=15)
        if r.status_code in (200, 201):
            print(f"✅ GitHub-т push хийгдлээ ({len(samples)} жишээ)")
            return True
        else:
            print(f"❌ GitHub push алдаа: {r.status_code} {r.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ GitHub push exception: {e}")
        return False


def add_feedback(sms_text: str, correct_category: str) -> List[Tuple[str, str]]:
    """Шинэ засалыг локал болон GitHub-т хадгалж бүх жагсаалтыг буцаана."""
    with _lock:
        samples = load_feedback()
        samples.append((sms_text, correct_category))
        _save_local(samples)
        # GitHub-т push хийх (амжилтгүй ч локал хадгалагдсан байна)
        _push_to_github(samples)
        return samples
