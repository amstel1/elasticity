import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Annotated, Optional, Literal
import uvicorn
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.context import CryptContext
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
import json
from loguru import logger
from datetime import datetime, timedelta
import pytz
import httpx
from pathlib import Path
import time

# --- Constants and Configuration ---
SESSION_FILES_DIR = Path("./session_data")
SESSION_FILES_DIR.mkdir(exist_ok=True)
DATABASE_URL = "sqlite:///./elasticity_model.db"
SECRET_KEY = "your_secret_key" # Replace with a strong, randomly generated secret key in production
FEATURES_API_URL = "http://localhost:8002/features"
PREDICT_API_URL = "http://localhost:8002/predict"
LOGIN_ERROR_URL = "/login?error=Incorrect username or password"
LOGIN_URL = "/login"
ROOT_URL = "/"

HEADER_COLUMNS = (
    "nomer_punkta",
    "usd__kurs_prinyato", "usd__Сумма принято", "usd__Курс перекрытия", "usd__Сумма выдано", "usd__kurs_vydano", "usd__Финансовый результат",
    "eur__kurs_prinyato", "eur__Сумма принято", "eur__Курс перекрытия", "eur__Сумма выдано", "eur__kurs_vydano", "eur__Финансовый результат",
    "rub__kurs_prinyato", "rub__Сумма принято", "rub__Курс перекрытия", "rub__Сумма выдано", "rub__kurs_vydano", "rub__Финансовый результат",
)

# --- Globals ---
cached_features: Dict = {'timestamp': None, 'features': {}}

# --- Database Setup ---
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# --- Security ---
PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="/token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return PASSWORD_CONTEXT.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)

# --- Data Models ---
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_name: str = Field(unique=True, index=True)
    hashed_password: str
    disabled: bool = Field(default=False)

class UserCreate(SQLModel):
    id: int | None
    user_name: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class Prediction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_id: int
    model_name: str
    inserted_datetime: datetime = Field(default_factory=lambda: pytz.timezone('Europe/Minsk').localize(datetime.now()))
    nomer_punkta: int
    hour: int
    kurs: float
    t: float
    y: float

# --- Database Utility Functions ---
def get_db():
    with Session(engine) as session:
        yield session

def insert_predictions_to_db(db: Session, dataloads: List[Dict]):
    """Inserts multiple prediction records into the database."""
    predictions = [Prediction(**prediction_kwargs) for prediction_kwargs in dataloads]
    db.add_all(predictions)
    db.commit()

# --- Session Management ---

middleware = [
    Middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY,
        session_cookie="session",  # Changed session_cookie to cookie_name
        https_only=False, # use https_only instead of cookie_https_only
        max_age=60 * 60 * 24 * 7  # Added max_age for session expiry (optional)
    )
]

# --- FastAPI App Setup ---
app = FastAPI(middleware=middleware)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Sample Data (Consider removing in production) ---


# --- Event Handlers ---
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# --- Dependency Functions ---
async def get_current_user(
        request: Request, db: Session = Depends(get_db)
) -> Optional[User]:
    """Retrieves the current user based on the session token."""
    user_name = request.session.get("access_token")
    if not user_name:
        return None
    user = db.exec(select(User).where(User.user_name == user_name)).first()
    return user

async def get_current_active_user(
    request: Request, current_user: Annotated[Optional[User], Depends(get_current_user)]
) -> User:
    """Retrieves the current active user, raising an exception if not authenticated or inactive."""
    if not current_user:
        request.session["redirect_url"] = str(request.url)
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": LOGIN_URL},
            detail="Not authenticated",
        )
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

# --- API Interaction Functions ---
async def fetch_features() -> Dict:
    """Fetches features from the external API, using a cache."""
    timestamp = cached_features.get('timestamp')
    if timestamp and datetime.now() - pd.Timestamp(timestamp).to_pydatetime() < timedelta(hours=2, minutes=55):
        return cached_features['features']

    async with httpx.AsyncClient() as client:
        response = await client.get(FEATURES_API_URL)
        response.raise_for_status()
        features = response.json()
        cached_features['features'] = features
        cached_features['timestamp'] = str(datetime.now())
        logger.debug(f'Fetched features from API. Cache timestamp: {cached_features["timestamp"]}')
        return features

async def fetch_predictions_from_api() -> List[Dict]:
    """Fetches predictions from the prediction API."""
    # logger.debug(f"Sending features to prediction API: {features}")
    async with httpx.AsyncClient() as client:
        # response = await client.post(PREDICT_API_URL, json={"features": features})
        # response.raise_for_status()
        # predictions = response.json()

        predictions = [
            {"id": 700, "nomer_punkta": 700, "usd__kurs_prinyato": 3.40, "usd__kurs_vydano": 3.45,
             "usd__Сумма принято": 4500, "usd__Сумма выдано": 5000, "usd__Финансовый результат": 237.5,
             "usd__Курс перекрытия": 3.425, "usd__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                                                                "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}},
             "eur__kurs_prinyato": 3.40, "eur__kurs_vydano": 3.45, "eur__Сумма принято": 4500,
             "eur__Сумма выдано": 5000, "eur__Финансовый результат": 237.5, "eur__Курс перекрытия": 3.425,
             "eur__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}},
             "rub__kurs_prinyato": 3.40, "rub__kurs_vydano": 3.45, "rub__Сумма принято": 4500,
             "rub__Сумма выдано": 5000, "rub__Финансовый результат": 237.5, "rub__Курс перекрытия": 3.425,
             "rub__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},

            {"id": 701, "nomer_punkta": 701, "usd__kurs_prinyato": 3.40, "usd__kurs_vydano": 3.45,
             "usd__Сумма принято": 4500,
             "usd__Сумма выдано": 5000, "usd__Финансовый результат": 237.5, "usd__Курс перекрытия": 3.425,
             "usd__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}},
             "eur__kurs_prinyato": 3.40, "eur__kurs_vydano": 3.45, "eur__Сумма принято": 4500,
             "eur__Сумма выдано": 5000,
             "eur__Финансовый результат": 237.5, "eur__Курс перекрытия": 3.425,
             "eur__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}},
             "rub__kurs_prinyato": 3.40, "rub__kurs_vydano": 3.45, "rub__Сумма принято": 4500,
             "rub__Сумма выдано": 5000,
             "rub__Финансовый результат": 237.5, "rub__Курс перекрытия": 3.425,
             "rub__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},
        ]
        logger.info(f"Received predictions from API: {predictions}")
        return predictions

# --- Route Handlers ---
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return PlainTextResponse(status_code=204)

@app.get(LOGIN_URL, response_class=HTMLResponse, name="login")
async def login_form(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/token", response_class=RedirectResponse)
async def login_process(
        request: Request,
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        db: Session = Depends(get_db)
):
    user = db.exec(select(User).where(User.user_name == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        return RedirectResponse(url=LOGIN_ERROR_URL, status_code=status.HTTP_303_SEE_OTHER)

    request.session["access_token"] = user.user_name
    return RedirectResponse(url=ROOT_URL, status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout", name="logout")
async def logout(request: Request):
    request.session.clear()
    response = RedirectResponse(url=LOGIN_URL, status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.exec(select(User).where(User.user_name == user.user_name)).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(id=user.id, user_name=user.user_name, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}

@app.get(ROOT_URL, response_class=HTMLResponse, name="list_view")
async def list_view(
        request: Request,
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    request.session['last_page'] = ROOT_URL
    form_data = request.session.get("form_data", {})
    if isinstance(form_data, str):
        form_data = json.loads(form_data)

    items_data = request.session.get("items", [])
    if not items_data:
        # init load
        items_data = await fetch_predictions_from_api()
        request.session["items"] = items_data
    logger.info(f'list view: {len(items_data)}, {type(items_data)}, {items_data}')
    if current_user.id != 369:
        items_data = [item for item in items_data if int(item.get('nomer_punkta', 99999) // 100) == current_user.id]

    return templates.TemplateResponse("list.html", {"request": request, "items": items_data, "current_user": current_user,
                                                    "form_data": form_data, "header_columns": HEADER_COLUMNS})

@app.get("/{item_id}/", response_class=HTMLResponse, name="detail_view")
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    request.session['last_page'] = f"/{item_id}/"
    form_data = request.session.get("form_data", {})
    if isinstance(form_data, str):
        try:
            form_data = json.loads(form_data)
            # logger.debug(f"Form data in detail view for item {item_id}: {form_data}")
        except json.JSONDecodeError:
            form_data = {}

    items_data = request.session.get("items", [])
    if not items_data:
        # init load
        items_data = await fetch_predictions_from_api()
        request.session["items"] = items_data
    logger.info(f'detail view: {len(items_data)}, {type(items_data)}, {items_data}')
    detail_view_items = [item for item in items_data if item.get('nomer_punkta') == item_id]
    return templates.TemplateResponse("detail.html",
                                      {"request": request, "items": detail_view_items,
                                       "current_user": current_user, "form_data": form_data, "header_columns": HEADER_COLUMNS})

@app.post("/calculate", name="calculate")
async def calculate(
    request: Request,
    current_user: Annotated[User, Depends(get_current_active_user)],
    mode: Literal['fast', 'slow', 'explain'] = Form(...)
):
    form_data_new = await request.form()
    old_form_data_str = request.session.get("form_data", "{}")

    try:
        old_form_data = json.loads(old_form_data_str)
    except json.JSONDecodeError:
        # logger.warning("Invalid JSON in old_form_data, resetting.")
        old_form_data = {}

    updated_form_data = old_form_data.copy()
    updated_form_data.update(form_data_new)
    request.session["form_data"] = json.dumps(updated_form_data)

    # logger.debug(f"Merged form data: {updated_form_data}")

    features = None
    if mode == 'slow':
        features = await fetch_features()
    elif mode == 'fast':
        features = await fetch_features()
    elif cached_features.get('features'):
        features = cached_features['features']
    else:
        features = await fetch_features()

    predictions = await fetch_predictions_from_api()

    if mode == 'explain':
        pass  # Add explain logic here

    request.session["items"] = json.dumps(predictions)
    logger.info(f'calc: {request.session["items"] }')
    time.sleep(1)  # Simulate processing time
    logger.info(f'Predictions updated in session. Redirecting to: {request.session.get("last_page", ROOT_URL)}')
    return RedirectResponse(url=request.session.get("last_page", ROOT_URL), status_code=status.HTTP_303_SEE_OTHER)

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)