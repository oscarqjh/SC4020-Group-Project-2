"""
Simple FastAPI backend that exposes the CrewAI medical analysis system over HTTP
and serves a lightweight front-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from src.crew.crew_manager import create_medical_crew, MedicalAnalysisCrew


APP_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_ROOT / "frontend"

SYSTEM_INSTRUCTIONS = """
MEDICAL ANALYSIS AI SYSTEM
======================================================================

Welcome to the Medical Analysis AI System!
This system can help analyze:
• Disease symptoms and provide preliminary diagnosis suggestions
• Breast cancer patient data and tumor characteristics

Available capabilities:
• Disease Analysis Tool – Use for natural language symptom descriptions
• Breast Cancer Analysis Tool – Use for tumor measurements/characteristics
• Symptom Extraction Tool – Identifies symptoms inside free text
• Breast Cancer Feature Extraction Tool – Identifies tumor metrics inside free text

IMPORTANT DISCLAIMER:
This system provides informational analysis only.
Always consult healthcare professionals for medical decisions.
======================================================================
""".strip()


class SessionState:
    """Wrap each Crew session so symptom state does not leak between users."""

    def __init__(self) -> None:
        self.crew: MedicalAnalysisCrew = create_medical_crew()
        self.crew.initialize()

    def run(self, message: str) -> str:
        return self.crew.execute_task(message)


class SessionManager:
    """In-memory session manager for chat conversations."""

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = SessionState()
        return session_id

    def get(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("invalid session id")
        return session


session_manager = SessionManager()

app = FastAPI(title="Medical Analysis Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionCreateResponse(BaseModel):
    session_id: str
    instructions: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Active session id")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    response: str


class DiseaseChecklistRequest(BaseModel):
    session_id: str
    symptoms: list[str]


class BreastCancerRequest(BaseModel):
    session_id: str
    measurements: dict[str, float]


@app.post("/api/session", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    """Create a new CrewAI session for the chat interface."""
    session_id = session_manager.create_session()
    return SessionCreateResponse(session_id=session_id, instructions=SYSTEM_INSTRUCTIONS)


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_agent(payload: ChatRequest) -> ChatResponse:
    """Send a chat message to the CrewAI system."""
    try:
        session = session_manager.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = session.run(payload.message)
    return ChatResponse(response=response)


@app.post("/api/disease/checklist", response_model=ChatResponse)
def run_disease_from_checklist(payload: DiseaseChecklistRequest) -> ChatResponse:
    """Trigger the disease analysis tool from selected symptoms."""
    if not payload.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")

    try:
        session = session_manager.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    symptom_text = ", ".join(payload.symptoms)
    prompt = f"The patient reports the following symptoms: {symptom_text}."
    response = session.run(prompt)
    return ChatResponse(response=response)


@app.post("/api/breast/predict", response_model=ChatResponse)
def run_breast_cancer_analysis(payload: BreastCancerRequest) -> ChatResponse:
    """Trigger the breast cancer analysis tool based on structured inputs."""
    if not payload.measurements:
        raise HTTPException(status_code=400, detail="No measurements provided")

    try:
        session = session_manager.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    parts = [f"{k.replace('_', ' ')} {v}" for k, v in payload.measurements.items()]
    prompt = (
        "A breast cancer study was performed on a patient. "
        "Doctors scanned the sizes of the cancer cell nuclei and performed a detailed analysis on the tumor. "
        "Here are the report details and measurements: " + "; ".join(parts) + ". "
        "Analyze these tumor characteristics and measurements to provide a comprehensive breast cancer assessment."
    )
    response = session.run(prompt)
    return ChatResponse(response=response)


@app.get("/")
def serve_root() -> FileResponse:
    """Serve the SPA entry point."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend assets missing")
    return FileResponse(index_path)


app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIR),
    name="assets",
)
