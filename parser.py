import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedSMS:
    raw_text: str
    amount: Optional[int]
    merchant: Optional[str]
    balance: Optional[int]
    is_incoming: bool
    card_last4: Optional[str]

def parse_khan_bank_sms(sms_text: str) -> ParsedSMS:
    text = sms_text.strip()
    
    # 1. Дүн олох (ZARLAGA: 68,000.00 эсвэл 25,000₮)
    amount = None
    amount_match = re.search(r'(?:ZARLAGA|ORLOGO):\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if not amount_match:
        amount_match = re.search(r'([\d,]+\.?\d*)\s*(?:₮|MNT|төгрөг)', text, re.IGNORECASE)
    
    if amount_match:
        # Цэг болон таслалыг цэвэрлэж бүхэл тоо болгох
        clean_amount = amount_match.group(1).replace(',', '')
        amount = int(float(clean_amount))

    # 2. Үлдэгдэл олох (ULDEGDEL:14,171.70)
    balance = None
    balance_match = re.search(r'ULDEGDEL:\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if balance_match:
        clean_balance = balance_match.group(1).replace(',', '')
        balance = int(float(clean_balance))

    # 3. Мерчант / Гүйлгээний утга
    merchant = None
    # 'Guilgeenii utga:' эсвэл 'Худалдаа:' гэдгийн ард байгааг авах
    merchant_match = re.search(r'(?:Guilgeenii utga|Худалдаа|Гүйлгээ|Хүлээн авагч):\s*([^\.\n]+)', text, re.IGNORECASE)
    
    if merchant_match:
        full_utga = merchant_match.group(1).strip()
        # TRF=... кодын ард байгаа цэвэр нэрийг салгах
        if '-' in full_utga:
            merchant = full_utga.split('-')[-1].strip()
        else:
            merchant = full_utga

    # 4. Орлого уу, Зарлага уу?
    is_incoming = bool(re.search(r'(ORLOGO|ирлээ|орлоо|нэмэгдлээ)', text, re.IGNORECASE))

    # 5. Карт/Дансны сүүлийн 4 орон
    card_match = re.search(r'(\d\*+\d{4})', text)
    card_last4 = card_match.group(1)[-4:] if card_match else None

    return ParsedSMS(
        raw_text=text,
        amount=amount,
        merchant=merchant,
        balance=balance,
        is_incoming=is_incoming,
        card_last4=card_last4
    )