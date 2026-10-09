"""
FastAPI Application

Main FastAPI app for JARVIS API service.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config.logging import setup_logging
from core.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger = get_settings().__class__.__module__.split(".")[0]
    logging.getLogger(__name__).info("Starting JARVIS API")
    yield
    logging.getLogger(__name__).info("Shutting down JARVIS API")


app = FastAPI(
    title="JARVIS API",
    description="Personal AI Assistant for Windows",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "jarvis"}


import logging