import requests
query = "hi"
url = f"http://127.0.0.1:5000/chat/{query}"
data = {
    "logs": """[2025-04-02 12:30:45] ERROR: Database connection failed.
               [2025-04-02 12:31:00] WARNING: High memory usage detected.
               [2025-04-02 12:32:15] INFO: User login successful."""
}

response = requests.post(url, json=data)
print(response.json())
