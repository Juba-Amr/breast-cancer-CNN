from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    with open("unit_tests/image/img_9225.png", "rb") as f:
        response = client.post("/", files={"file": f})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 0

def test_predict_endpoint_missing_file():
    response = client.post("/")  
    assert response.status_code == 422