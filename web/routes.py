"""FastAPI routes - pages and form handlers."""

import re
from pathlib import Path
from fastapi import APIRouter, Request, Depends, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from db import crud
from db.models import User
from .auth import create_session_cookie
from .dependencies import get_db, get_current_user
from src.graph import run_query
from src.config import Config
from src.ingestion.document_parser import DocumentParser
from src.ingestion.user_embedder import UserEmbedder
from src.ingestion.llm_extractor import LLMExtractor, merge_extracted_data

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")


def _extract_profile_from_text(profile: dict, text: str) -> tuple[dict, bool]:
    """Extract income/expenses from chat text and update profile. Returns (profile, was_updated)."""
    updated = False
    text_lower = text.lower()

    amount = r"(?:₹|rs\.?|inr)?\s*(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>l|lac|lakh|k)?"
    joiner = r"(?:\s+(?:is|are|am|around|about|roughly|approx|approximately|of|as))*\s+"
    income_words = r"income|salary|earn|earning|make|making"
    expense_words = r"expense|expenses|spend|spending|expenditure|cost|costs"

    income_patterns = [
        rf"\b(?:{income_words})\b{joiner}{amount}",
    ]
    expense_patterns = [
        rf"\b(?:{expense_words})\b{joiner}{amount}",
        rf"{amount}\s*(?:per\s+month|/month|monthly)?\s+\b(?:{expense_words})\b",
    ]

    def _parse_amount(match):
        val = float(match.group("value").replace(",", ""))
        suffix = match.group("suffix") or ""
        if suffix in ("l", "lac", "lakh"):
            val *= 100000
        elif suffix == "k":
            val *= 1000
        return int(val)

    def _is_annual_income(match):
        window = text_lower[match.end():match.end() + 30]
        return bool(re.search(r"\b(?:per\s+year|per\s+annum|annually|annual|yearly|pa|p\.a\.)\b", window))

    def _first_match(patterns):
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match
        return None

    inc_match = _first_match(income_patterns)
    exp_match = _first_match(expense_patterns)

    if inc_match:
        income_amount = _parse_amount(inc_match)
        profile.setdefault("income", {})
        if _is_annual_income(inc_match):
            profile["income"]["annual"] = income_amount
            profile["income"]["monthly"] = income_amount // 12
        else:
            profile["income"]["monthly"] = income_amount
            profile["income"]["annual"] = income_amount * 12
        updated = True

    if exp_match:
        profile.setdefault("expenses", {})["monthly"] = _parse_amount(exp_match)
        updated = True

    if re.search(r"risk.?tak|aggressive|high.{0,15}risk|risk.{0,15}high", text_lower):
        profile["risk_tolerance"] = "aggressive"
        updated = True
    elif re.search(r"moderate|balanced|medium.{0,15}risk|risk.{0,15}medium", text_lower):
        profile["risk_tolerance"] = "moderate"
        updated = True
    elif re.search(r"conservative|safe|low.{0,15}risk|risk.{0,15}low|no risk|risk.?averse", text_lower):
        profile["risk_tolerance"] = "conservative"
        updated = True

    return profile, updated



@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_email(db, email)
    if not user or user.password != password:
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid email or password"}
        )

    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("session", create_session_cookie(user.id), httponly=True)
    return response


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html", {"error": None})


@router.post("/signup")
def signup_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    if crud.get_user_by_email(db, email):
        return templates.TemplateResponse(
            request, "signup.html", {"error": "Email already registered"}
        )

    user = crud.create_user(db, email, password)
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("session", create_session_cookie(user.id), httponly=True)
    return response


@router.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response



@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    cookie = request.cookies.get("session")
    if cookie:
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request, "home.html", {})



@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    profile = crud.get_profile(db, user.id)
    messages = crud.get_chat_history(db, user.id)
    documents = crud.list_uploaded_documents(db, user.id)

    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user,
        "profile": profile,
        "messages": messages,
        "documents": documents,
    })



@router.post("/chat")
def chat_submit(
    request: Request,
    query: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    uid_str = crud.user_id_str(user.id)
    profile = crud.get_profile(db, user.id)

    profile, updated = _extract_profile_from_text(profile, query)
    if updated:
        crud.update_profile(db, user.id, profile)

    crud.add_chat_message(db, user.id, "user", query)

    result = run_query(query, user_id=uid_str, user_profile=profile)

    recommendation = result.get("recommendation", "No recommendation generated")
    metadata = {
        "confidence": result.get("confidence", 0.5),
        "calculations": result.get("calculations", []),
        "tool_calls": result.get("tool_calls", []),
        "errors": result.get("errors", []),
    }

    crud.add_chat_message(db, user.id, "assistant", recommendation, metadata)

    return RedirectResponse("/dashboard", status_code=303)



@router.get("/profile", response_class=HTMLResponse)
def profile_page(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    profile = crud.get_profile(db, user.id)
    return templates.TemplateResponse(request, "profile.html", {
        "user": user,
        "profile": profile,
    })


@router.post("/profile")
def profile_submit(
    request: Request,
    monthly_income: str = Form(""),
    income_source: str = Form(""),
    monthly_expenses: str = Form(""),
    risk_tolerance: str = Form(""),
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    profile = crud.get_profile(db, user.id)

    if monthly_income.strip():
        val = int(float(monthly_income))
        profile["income"]["monthly"] = val
        profile["income"]["annual"] = val * 12
    if income_source.strip():
        profile["income"]["source"] = income_source
    if monthly_expenses.strip():
        profile["expenses"]["monthly"] = int(float(monthly_expenses))
    if risk_tolerance.strip():
        profile["risk_tolerance"] = risk_tolerance

    crud.update_profile(db, user.id, profile)
    return RedirectResponse("/dashboard", status_code=303)



@router.post("/upload")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    uid_str = crud.user_id_str(user.id)
    docs_dir = Config.get_user_documents_path(uid_str)
    dest_path = docs_dir / file.filename

    with open(dest_path, "wb") as f:
        f.write(file.file.read())

    embedder = UserEmbedder()
    num_chunks = embedder.ingest_user_document(uid_str, dest_path)

    crud.add_uploaded_document(db, user.id, file.filename, Path(file.filename).suffix, num_chunks)

    if num_chunks > 0:
        try:
            parser = DocumentParser()
            chunks = parser.parse_file(dest_path)
            doc_text = "\n".join(c.content for c in chunks)

            extractor = LLMExtractor()
            extracted = extractor.extract(doc_text)

            if "error" not in extracted:
                extracted.pop("document_summary", None)
                profile = crud.get_profile(db, user.id)
                updated = merge_extracted_data(profile, extracted)
                crud.update_profile(db, user.id, updated)
        except Exception:
            pass  # Extraction is best-effort

    return RedirectResponse("/dashboard", status_code=303)


@router.post("/documents/{filename}/delete")
def delete_document(
    filename: str,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    uid_str = crud.user_id_str(user.id)
    embedder = UserEmbedder()
    embedder.delete_user_document(uid_str, filename)
    crud.delete_uploaded_document(db, user.id, filename)

    return RedirectResponse("/dashboard", status_code=303)



@router.post("/chat/clear")
def clear_chat(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse("/login", status_code=303)

    crud.clear_chat_history(db, user.id)
    return RedirectResponse("/dashboard", status_code=303)
