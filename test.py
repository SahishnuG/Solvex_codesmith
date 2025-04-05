import requests

# 1. First add logs to the vector database
add_logs_response = requests.post(
    "http://127.0.0.1:5000/add_logs", 
    json={
        "logs": """
# Order Service
2023-11-15T10:15:22.456Z INFO [order-service] - Received new order
trace_id=7d3a4b1c2f8e5d6a9c0b1a2d3e4f5g6h
span_id=a1b2c3d4e5f6g7h8
user_id=7890
order_id=ORD-54321
component=order_api
{ "items": 4, "total": 89.95 }
# Payment Service
2023-11-15T10:15:23.123Z INFO [payment-service] - Processing payment
trace_id=7d3a4b1c2f8e5d6a9c0b1a2d3e4f5g6h
span_id=b2c3d4e5f6g7h8i9
parent_span_id=a1b2c3d4e5f6g7h8
user_id=7890
order_id=ORD-54321
component=payment_processor
{ "amount": 89.95, "method": "credit_card" }
# Inventory Service
2023-11-15T10:15:24.789Z INFO [inventory-service] - Reserved inventory
trace_id=7d3a4b1c2f8e5d6a9c0b1a2d3e4f5g6h
span_id=c3d4e5f6g7h8i9j0
parent_span_id=a1b2c3d4e5f6g7h8
order_id=ORD-54321
component=inventory_reserver
{ "skus": ["PROD-123", "PROD-456"], "quantities": [2, 2] }"""
    }
)
print("Add logs response:", add_logs_response.json())

# 2. Then analyze the logs from the vector database

analyse = requests.post(
    "http://127.0.0.1:5000/analyze_logs",
    json={
        "query":"Find patterns in the given logs, especially to predict possible future issues",
        "limit":10
    }
)
print(f"Analysis: {analyse.json()}")
print("-"*50+"\n")
# 3. Chat
chat = requests.post(
    "http://127.0.0.1:5000/chat/hi", 
    json={
        "query": "Find patterns in the given logs, especially to predict possible future issues",
        "limit": 10
    }
)
print("Chat response:", chat.json())