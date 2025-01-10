from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib  # Or pickle, depending on how you saved your model

# Define the data model for the input JSON
class InputData(BaseModel):
    # Replace these with the actual feature names and their data types
    feature1: float
    feature2: int
    feature3: str
    # ... add all your model's input features here

# Initialize the FastAPI app
app = FastAPI()

# Global variable to store the loaded model
model = None

# Event handler to load the model on startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("your_model.joblib")  # Replace with your actual model file path
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading the model: {e}")
        # It's good practice to raise an exception to prevent the app from starting
        raise  # or you can implement fallback behavior

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    """
    Make predictions using the loaded gradient boosting model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check the server logs.")

    try:
        # Convert the input data to a Pandas DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Make the prediction
        prediction = model.predict(input_df)

        # Return the prediction as JSON
        return {"prediction": prediction.tolist()}  # Convert NumPy array to list for JSON serialization

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)