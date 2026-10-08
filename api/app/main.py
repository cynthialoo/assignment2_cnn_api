import os
from typing import Any

import spacy
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.cnn_model import CIFAR10_CLASSES
from app.image_classifier import DEVICE, MODEL_LOADED, MODEL_PATH, MODEL_WARNING, classify_upload


MODEL_NAME = os.getenv("SPACY_MODEL", "en_core_web_lg")


try:
    nlp = spacy.load(MODEL_NAME)
except OSError as exc:
    raise RuntimeError(
        f"spaCy model '{MODEL_NAME}' is not installed. "
        f"Install it with: python -m spacy download {MODEL_NAME}"
    ) from exc


app = FastAPI(
    title="Assignment 2 CNN Image Classifier API",
    description=(
        "FastAPI service that keeps the Assignment 1 embedding endpoints "
        "and adds a CIFAR-10 CNN image classifier for Assignment 2."
    ),
    version="2.0.0",
)


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, description="Query word to embed")


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, description="First query word")
    word2: str = Field(..., min_length=1, description="Second query word")


def _clean_word(word: str) -> str:
    cleaned = word.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="The query word cannot be empty.")
    return cleaned


def _doc_for_word(word: str) -> Any:
    cleaned = _clean_word(word)
    doc = nlp(cleaned)
    if len(doc) == 0:
        raise HTTPException(status_code=422, detail="The query word could not be parsed.")
    if not doc.has_vector or doc.vector_norm == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No spaCy vector was found for '{cleaned}'. Try another English word."
            ),
        )
    return doc


def _embedding_response(word: str) -> dict[str, Any]:
    doc = _doc_for_word(word)
    vector = [float(value) for value in doc.vector]
    return {
        "word": word.strip(),
        "model": MODEL_NAME,
        "dimension": len(vector),
        "has_vector": doc.has_vector,
        "embedding": vector,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Assignment 2 CNN Image Classifier API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "spacy_model": MODEL_NAME,
        "cnn_device": str(DEVICE),
        "cnn_model_path": str(MODEL_PATH),
        "cnn_model_loaded": MODEL_LOADED,
        "cnn_warning": MODEL_WARNING,
    }


@app.get("/cifar10/classes")
def cifar10_classes() -> dict[str, Any]:
    return {"classes": CIFAR10_CLASSES}


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)) -> dict[str, Any]:
    return await classify_upload(file)


@app.get("/embedding")
def get_embedding(
    word: str = Query(..., min_length=1, description="Query word to embed")
) -> dict[str, Any]:
    return _embedding_response(word)


@app.post("/embedding")
def post_embedding(request: EmbeddingRequest) -> dict[str, Any]:
    return _embedding_response(request.word)


@app.get("/similarity")
def get_similarity(
    word1: str = Query(..., min_length=1, description="First query word"),
    word2: str = Query(..., min_length=1, description="Second query word"),
) -> dict[str, Any]:
    doc1 = _doc_for_word(word1)
    doc2 = _doc_for_word(word2)
    return {
        "word1": word1.strip(),
        "word2": word2.strip(),
        "model": MODEL_NAME,
        "similarity": float(doc1.similarity(doc2)),
    }


@app.post("/similarity")
def post_similarity(request: SimilarityRequest) -> dict[str, Any]:
    return get_similarity(request.word1, request.word2)
