"""Tests for the FastAPI REST endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.app import app
    with TestClient(app) as c:
        yield c


class TestHealthRoutes:

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_endpoint_returns_checks(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data


class TestMCPRoutes:

    def test_list_mcp_tools(self, client):
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        tool_names = [t["name"] for t in data["tools"]]
        assert "search_documents" in tool_names
        assert "ask_question" in tool_names

    def test_list_mcp_resources(self, client):
        response = client.get("/mcp/resources")
        assert response.status_code == 200
        data = response.json()
        assert "resources" in data
        uris = [r["uri"] for r in data["resources"]]
        assert any("documents" in uri for uri in uris)

    def test_list_mcp_prompts(self, client):
        response = client.get("/mcp/prompts")
        assert response.status_code == 200
        data = response.json()
        assert "prompts" in data
        prompt_names = [p["name"] for p in data["prompts"]]
        assert "rag_answer" in prompt_names
        assert "document_analysis" in prompt_names

    def test_get_rag_answer_prompt(self, client):
        response = client.post(
            "/mcp/prompts/get",
            json={
                "prompt": "rag_answer",
                "arguments": {
                    "question": "What is RAG?",
                    "context": "RAG is Retrieval-Augmented Generation.",
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) > 0

    def test_get_nonexistent_prompt_returns_404(self, client):
        response = client.post(
            "/mcp/prompts/get",
            json={"prompt": "nonexistent_prompt_xyz", "arguments": {}},
        )
        assert response.status_code == 404
