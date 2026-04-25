from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from classifier import classifier, CATEGORY_ICONS
from parser import parse_khan_bank_sms


def is_meaningful_merchant(merchant) -> bool:
    """Худалдаачин/гүйлгээний утга утга төгөлдөр эсэхийг шалгах.
    'a', 'b', '1', '12', 'aa' гэх мэт богино, утгагүй текст байвал False."""
    if not merchant:
        return False
    cleaned = merchant.strip()
    # 3 тэмдэгтээс богино бол утгагүй
    if len(cleaned) < 3:
        return False
    # Зөвхөн тоо/тэмдэгтээс бүрдсэн бол утгагүй (жишээ: "123", "***")
    if not any(c.isalpha() for c in cleaned):
        return False
    return True


def resolve_category(prediction: dict, merchant) -> dict:
    """Хэрэв гүйлгээний утга утгагүй бол 'Бусад' гэж ангилах."""
    if not is_meaningful_merchant(merchant):
        return {
            "category": "Бусад",
            "icon": CATEGORY_ICONS.get("Бусад", "❓"),
            "confidence": prediction.get("confidence", 0.0),
        }
    return prediction

app = FastAPI(
    title="SMS Зарлага Ангилагч API",
    description="Хаан банкны SMS-ийг ML ашиглан ангилах API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Апп эхлэхэд model-ийг load хийнэ
@app.on_event("startup")
async def startup_event():
    classifier.load()
    print("✅ SMS Classifier API started")


# ── Request / Response Models ──────────────────────────────────────────────

class SMSRequest(BaseModel):
    sms_text: str
    timestamp: Optional[str] = None  # ISO format: "2024-01-15T12:30:00"

class BatchSMSRequest(BaseModel):
    messages: List[SMSRequest]

class FeedbackRequest(BaseModel):
    sms_text: str
    correct_category: str

class TransactionResponse(BaseModel):
    sms_text: str
    amount: Optional[int]
    merchant: Optional[str]
    balance: Optional[int]
    is_incoming: bool
    card_last4: Optional[str]
    category: str
    icon: str
    confidence: float
    timestamp: Optional[str]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SMS Зарлага Ангилагч API ажиллаж байна 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": classifier.is_trained}

@app.post("/classify", response_model=TransactionResponse)
def classify_sms(request: SMSRequest):
    """Нэг SMS-ийг ангилна"""
    try:
        parsed = parse_khan_bank_sms(request.sms_text)
        prediction = classifier.predict(request.sms_text)
        prediction = resolve_category(prediction, parsed.merchant)

        return TransactionResponse(
            sms_text=request.sms_text,
            amount=parsed.amount,
            merchant=parsed.merchant,
            balance=parsed.balance,
            is_incoming=parsed.is_incoming,
            card_last4=parsed.card_last4,
            category=prediction["category"],
            icon=prediction["icon"],
            confidence=prediction["confidence"],
            timestamp=request.timestamp or datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=List[TransactionResponse])
def classify_batch(request: BatchSMSRequest):
    """Олон SMS-ийг нэгэн зэрэг ангилна"""
    results = []
    for sms_req in request.messages:
        try:
            parsed = parse_khan_bank_sms(sms_req.sms_text)
            prediction = classifier.predict(sms_req.sms_text)
            prediction = resolve_category(prediction, parsed.merchant)
            results.append(TransactionResponse(
                sms_text=sms_req.sms_text,
                amount=parsed.amount,
                merchant=parsed.merchant,
                balance=parsed.balance,
                is_incoming=parsed.is_incoming,
                card_last4=parsed.card_last4,
                category=prediction["category"],
                icon=prediction["icon"],
                confidence=prediction["confidence"],
                timestamp=sms_req.timestamp or datetime.now().isoformat(),
            ))
        except Exception:
            continue
    return results

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Хэрэглэгч ангиллыг засах үед model-ийг дахин сургана"""
    valid_categories = list(CATEGORY_ICONS.keys())
    if request.correct_category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Буруу категори. Зөв категориуд: {valid_categories}"
        )
    classifier.add_training_sample(request.sms_text, request.correct_category)
    return {"message": f"✅ Баярлалаа! Model '{request.correct_category}' категориор шинэчлэгдлээ."}

@app.get("/categories")
def get_categories():
    """Нийт категориудын жагсаалт"""
    return [
        {"name": name, "icon": icon}
        for name, icon in CATEGORY_ICONS.items()
    ]

@app.post("/train")
def retrain_model():
    """Model-ийг дахин сургах"""
    accuracy = classifier.train()
    return {"message": "✅ Model амжилттай сургагдлаа", "accuracy": round(accuracy, 4)}