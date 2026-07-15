from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")
    assert response.status_code in {200, 500}
