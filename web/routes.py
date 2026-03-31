"""FastAPI routes — pages and form handlers."""

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


# ── Auth Pages ──

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


# ── Root ──

@router.get("/")
def root(request: Request):
    cookie = request.cookies.get("session")
    if cookie:
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


# ── Dashboard ──

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


# ── Chat ──

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

    # Save user message
    crud.add_chat_message(db, user.id, "user", query)

    # Run the AI pipeline with the user's profile
    result = run_query(query, user_id=uid_str, user_profile=profile)

    recommendation = result.get("recommendation", "No recommendation generated")
    metadata = {
        "confidence": result.get("confidence", 0.5),
        "calculations": result.get("calculations", []),
    }

    # Save assistant message
    crud.add_chat_message(db, user.id, "assistant", recommendation, metadata)

    return RedirectResponse("/dashboard", status_code=303)


# ── Profile ──

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

    # Update fields (only if provided)
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


# ── Document Upload ──

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

    # Save file to disk
    with open(dest_path, "wb") as f:
        f.write(file.file.read())

    # Parse & embed into user's FAISS index
    embedder = UserEmbedder()
    num_chunks = embedder.ingest_user_document(uid_str, dest_path)

    # Record in DB
    crud.add_uploaded_document(db, user.id, file.filename, Path(file.filename).suffix, num_chunks)

    # Extract structured data with LLM and update profile
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


# ── Clear Chat ──

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
