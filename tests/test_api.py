
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ============================================================
# 1. PROJECT IMPORT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.main import app


# ============================================================
# 2. TEST CLIENT
# ============================================================

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ============================================================
# 3. HOME ENDPOINT TEST
# ============================================================

def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["version"] == "2.1.0"

    assert "northstar" in data["available_datasets"]
    assert "apex" in data["available_datasets"]

    assert "FAISS Dense Retrieval" in data["architecture"]
    assert "BM25 Sparse Retrieval" in data["architecture"]
    assert "Reciprocal Rank Fusion" in data["architecture"]
    assert "Cross-Encoder Reranking" in data["architecture"]

    assert (
        "Evidence-Based Answerability Verification"
        in data["architecture"]
    )

    assert (
        "Complete Source Page Reconstruction"
        in data["architecture"]
    )


# ============================================================
# 4. SUPPORTED QUESTION TEST
# ============================================================

def test_supported_question(client):
    response = client.post(
        "/ask",
        json={
            "question": (
                "How many annual leave days do employees receive?"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["dataset"] == "northstar"

    assert "20 days" in data["answer"]

    assert data["evidence_supported"] is True

    assert len(data["verification_reason"]) > 0

    assert len(data["sources"]) > 0

    first_source = data["sources"][0]

    assert (
        first_source["source"]
        == "Leave_and_Attendance_Policy.pdf"
    )

    assert first_source["page"] == 1

    assert "rrf_score" in first_source
    assert "rerank_score" in first_source

    assert first_source["rrf_score"] is not None
    assert first_source["rerank_score"] is not None

    assert isinstance(first_source["rrf_score"], float)
    assert isinstance(first_source["rerank_score"], float)

    assert len(data["retrieved_sources"]) > 0


# ============================================================
# 5. UNSUPPORTED QUESTION TEST
# ============================================================

def test_unsupported_question(client):
    response = client.post(
        "/ask",
        json={
            "question": (
                "How many maternity leave days are provided?"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["evidence_supported"] is False

    assert (
        data["answer"]
        == "I could not find this information in the provided documents."
    )

    assert len(data["verification_reason"]) > 0


# ============================================================
# 6. INVALID REQUEST TEST
# ============================================================

def test_invalid_request(client):
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


# ============================================================
# 7. EMPTY QUESTION TEST
# ============================================================

def test_empty_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


# ============================================================
# 8. INVALID DATASET TEST
# ============================================================

def test_invalid_dataset(client):
    response = client.post(
        "/ask",
        json={
            "question": "What is the annual leave policy?",
            "dataset": "invalid_dataset",
        },
    )

    assert response.status_code == 400


# ============================================================
# 9. HEALTH ENDPOINT TEST
# ============================================================

def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["reranker_loaded"] is True
    assert "northstar" in data["loaded_datasets"]


