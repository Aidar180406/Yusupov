# test_api.py - тестирование API
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health:", response.json())

def test_predict():
    data = {
        "size": 120,
        "rooms_num": 3,
        "district": "Kartal",
        "building_age_num": 10,
        "total_floor_count_num": 20,
        "floor_category": "floor_5",
        "is_high_floor": 0,
        "is_penthouse": 0
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("Predict:", json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_districts():
    response = requests.get(f"{BASE_URL}/districts")
    print("Districts:", json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_health()
    test_districts()
    test_predict()