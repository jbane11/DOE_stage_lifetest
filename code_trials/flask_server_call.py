import requests

response = requests.get("http://127.0.0.1:5000/compute", params={"x": 3, "y": 7})
print(response.json())