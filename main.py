from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
import requests

app = FastAPI()

# Configure Jinja templates
templates = Jinja2Templates(directory="templates")

# Serve static files (for Pico CSS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Dummy data
items = [
    {"id": 1, "name": "Item A", "description": "Description for Item A"},
    {"id": 2, "name": "Item B", "description": "Description for Item B"},
    {"id": 3, "name": "Item C", "description": "Description for Item C"},
]

# POST_ENDPOINT = "http://localhost:8000/receive_data"  # Replace with your actual POST endpoint

@app.get("/", response_class=HTMLResponse)
async def list_view(request: Request):
    # Send a POST request on GET
    placeholder_data = {"action": "list_view_accessed"}
    try:
        pass
        # requests.post(POST_ENDPOINT, json=placeholder_data)
    except requests.exceptions.ConnectionError:
        # print(f"Warning: Could not connect to POST endpoint: {POST_ENDPOINT}")
        pass
    return templates.TemplateResponse("list.html", {"request": request, "items": items})

@app.get("/{item_id:int}/", response_class=HTMLResponse)
async def detail_view(request: Request, item_id: int):
    item = next((item for item in items if item["id"] == item_id), None)
    if not item:
        return HTMLResponse(content="Item not found", status_code=404)

    # Send a POST request on GET
    placeholder_data = {"action": f"detail_view_accessed", "item_id": item_id}
    try:
        pass
        # requests.post(POST_ENDPOINT, json=placeholder_data)
    except requests.exceptions.ConnectionError:
        pass
        # print(f"Warning: Could not connect to POST endpoint: {POST_ENDPOINT}")

    placeholder_value = "This is a placeholder value."
    return templates.TemplateResponse("detail.html", {"request": request, "item": item, "placeholder_value": placeholder_value})

# @app.post("/receive_data")
# async def receive_data(data: Dict):
#     print(f"Received POST data: {data}")
#     return {"status": "success", "message": "Data received"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)