"""FastAPI entry point for SentinelFinance."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from db.database import init_db
from web.routes import router

app = FastAPI(title="SentinelFinance")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(router)


@app.on_event("startup")
def startup():
    init_db()


# Run with: uvicorn main:app --reload
