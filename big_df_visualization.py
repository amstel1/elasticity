import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any

# --- App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Data Generation (for demonstration) ---
# In a real app, you would load this from a file or database.
def create_big_dataframe():
    """Creates a sample DataFrame."""
    data = {
        'ID': range(1, 1001),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
        'Value': np.random.uniform(10, 1000, size=1000),
        'Age': np.random.randint(18, 65, size=1000),
        'Rating': np.random.normal(5, 1.5, size=1000).round(2)
    }
    return pd.DataFrame(data)

# Load the data once when the app starts
df_original = create_big_dataframe()
NUMERICAL_COLS = df_original.select_dtypes(include=np.number).columns.tolist()

# --- FastAPI Endpoint ---

# Use @app.route to handle both GET and POST requests on the same URL
@app.route("/", methods=["GET", "POST"])
async def show_table(request: Request):
    df = df_original.copy()
    filter_values: Dict[str, Any] = {}

    if request.method == "POST":
        form_data = await request.form()
        action = form_data.get("action")

        if action == "filter":
            # Apply filters
            for col in NUMERICAL_COLS:
                min_val_str = form_data.get(f"{col}_min", "").strip()
                max_val_str = form_data.get(f"{col}_max", "").strip()
                
                # Store user input to re-populate the form
                filter_values[f"{col}_min"] = min_val_str
                filter_values[f"{col}_max"] = max_val_str

                try:
                    # Filter by min value if provided
                    if min_val_str:
                        min_val = float(min_val_str)
                        df = df[df[col] >= min_val]
                    # Filter by max value if provided
                    if max_val_str:
                        max_val = float(max_val_str)
                        df = df[df[col] <= max_val]
                except ValueError:
                    # Ignore invalid number inputs
                    pass
        # If action is "clear", we do nothing, effectively resetting to the original df
        # and an empty filter_values dictionary.

    # Prepare data for the template
    context = {
        "request": request,
        "title": "Pandas DataFrame Viewer",
        "columns": df.columns.tolist(),
        "numerical_cols": NUMERICAL_COLS,
        "data_rows": df.to_dict(orient="records"),
        "filter_values": filter_values,
    }
    return templates.TemplateResponse("index.html", context)
