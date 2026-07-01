"""FastAPI entry point for SentinelFinance."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from db.database import init_db
from web.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="SentinelFinance", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.include_router(router)
