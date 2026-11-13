
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Add project root to Python import path
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from api.main import app


@pytest.fixture(scope="module")
def client():

    with TestClient(app) as test_client:
        yield test_client


def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["version"] == "2.0.0"

    assert "BM25 Sparse Retrieval" in data["architecture"]

    assert "Cross-Encoder Reranking" in data["architecture"]

    assert (
        "Evidence-Based Answerability Verification"
        in data["architecture"]
    )


def test_supported_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "How many annual leave days do employees receive?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "20 days" in data["answer"]

    assert data["evidence_supported"] is True

    assert len(data["verification_reason"]) > 0

    assert len(data["sources"]) > 0

    first_source = data["sources"][0]

    assert first_source["source"] == "Leave_and_Attendance_Policy.pdf"

    assert first_source["page"] == 1

    assert "rrf_score" in first_source

    assert "rerank_score" in first_source


def test_unsupported_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "How many maternity leave days are provided?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["evidence_supported"] is False

    assert (
        data["answer"]
        == "I could not find this information in the provided documents."
    )

    assert len(data["verification_reason"]) > 0


def test_invalid_request(client):

    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422


def test_empty_question(client):

    response = client.post(
        "/ask",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422


