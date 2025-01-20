from pyexpat import features
from loguru import logger
from fastapi import FastAPI, HTTPException
from io import BytesIO
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib  # Or pickle, depending on how you saved your model
from typing import Dict, Any, Literal
import pickle
from postgres.ops import PostgresInterface

# Define the data model for the input JSON
class InputData(BaseModel):
    pass
    # Replace these with the actual feature names and their data types
    # feature1: float
    # feature2: int
    # feature3: str
    # ... add all your model's input features here

class GetDataInput(BaseModel):
    kind: Literal['voo', 'baseline_kurs', 'myfin']

class GetFeaturesInput(BaseModel):
    kind: Literal['voo', 'baseline_kurs', 'myfin']

# Initialize the FastAPI app
app = FastAPI()

# Global variable to store the loaded model
model = None

# Event handler to load the model on startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        # model = joblib.load("your_model.joblib")  # Replace with your actual model file path
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading the model: {e}")
        # It's good practice to raise an exception to prevent the app from starting
        raise  # or you can implement fallback behavior

@app.get("/get_data")
async def get_data(kind: GetDataInput):

    return {'vydano_usd': pd.DataFrame([{1: 'val1', 2: 'val2'}])}

# Form the features: time consuming. Results are to be cached.
# No params
@app.get("/get_features")
async def get_features(kind: GetFeaturesInput):
    # return: {task: pd.DataFrame}
    return {'vydano_usd': pd.DataFrame([{1: 'val1', 2: 'val2'}])}

@app.get("/explain")
async def explain(operation_type: Literal['prinyato','vydano'], this_currency: Literal['usd', 'eur', 'rub']):
    # insert here logic for creating a plot
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)  # Reset buffer position to the beginning
    return StreamingResponse(image_buffer, media_type="image/png")

@app.post("/predict")
async def predict(data: Dict[Any, Any]):
    logger.debug(data)
    """
    Make predictions using the loaded gradient boosting model.
    """
    # if model is None:
    #     raise HTTPException(status_code=500, detail="Model not loaded. Please check the server logs.")

    # try:
    #     # Convert the input data to a Pandas DataFrame
    #     input_df = pd.DataFrame([data.dict()])
    #
    #     # Make the prediction
    #     prediction = model.predict(input_df)
    #
    #     # Return the prediction as JSON
    #     return {"prediction": prediction.tolist()}  # Convert NumPy array to list for JSON serialization
    #
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    logger.debug("exit")
    return {'key': 'prediction_placeholder'}

@app.post("/optimize_profit")
async def optimize_profit(data: Dict[Any, Any]):
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)