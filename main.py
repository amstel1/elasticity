from http.client import responses

import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Annotated, Optional
import uvicorn
from fastapi import Depends
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware
from starlette.datastructures import FormData
import json
from loguru import logger
from datetime import datetime, timedelta
import pytz
from sqlmodel import Field, Session, SQLModel, create_engine, select
import httpx
from typing import Literal


global cached_features
cached_features = {'timestamp': None, 'features': {}}

header_columns = ("nomer_punkta",
    "usd__kurs_prinyato", "usd__Сумма принято", "usd__Курс перекрытия",  "usd__Сумма выдано", "usd__kurs_vydano", "usd__Финансовый результат",
    "eur__kurs_prinyato", "eur__Сумма принято", "eur__Курс перекрытия",  "eur__Сумма выдано", "eur__kurs_vydano", "eur__Финансовый результат",
    "rub__kurs_prinyato", "rub__Сумма принято", "rub__Курс перекрытия",  "rub__Сумма выдано", "rub__kurs_vydano", "rub__Финансовый результат",
                  )
DATABASE_URL = "sqlite:///./elasticity_model.db"
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="/token")
SECRET_KEY = "your_secret_key"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return PASSWORD_CONTEXT.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)


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

def insert_predictions( db: Session, dataloads = List[Dict],):
    """Inserts a new item into the database."""
    predictions = [Prediction(**prediction_kwargs) for prediction_kwargs in dataloads]
    db.add_all(predictions)
    db.commit()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

items = [
    {"id": 700, "nomer_punkta": 700, "usd__kurs_prinyato": 3.40, "usd__kurs_vydano": 3.45, "usd__Сумма принято": 4500, "usd__Сумма выдано": 5000, "usd__Финансовый результат": 237.5, "usd__Курс перекрытия": 3.425, "usd__Конкуренты":{"Альфа-Банк":{"Покупка": 3.41, "Продажа": 3.44}, "ВТБ-Банк":{"Покупка": 3.411, "Продажа": 3.439}},
     "eur__kurs_prinyato": 3.40, "eur__kurs_vydano": 3.45,  "eur__Сумма принято": 4500, "eur__Сумма выдано": 5000, "eur__Финансовый результат": 237.5, "eur__Курс перекрытия": 3.425, "eur__Конкуренты":{"Альфа-Банк":{"Покупка": 3.41, "Продажа": 3.44}, "ВТБ-Банк":{"Покупка": 3.411, "Продажа": 3.439}},
    "rub__kurs_prinyato": 3.40, "rub__kurs_vydano": 3.45,  "rub__Сумма принято": 4500, "rub__Сумма выдано": 5000, "rub__Финансовый результат": 237.5, "rub__Курс перекрытия": 3.425, "rub__Конкуренты":{"Альфа-Банк":{"Покупка": 3.41, "Продажа": 3.44}, "ВТБ-Банк":{"Покупка": 3.411, "Продажа": 3.439}}},

    {"id": 701, "nomer_punkta": 701, "usd__kurs_prinyato": 3.40, "usd__kurs_vydano": 3.45, "usd__Сумма принято": 4500,
     "usd__Сумма выдано": 5000, "usd__Финансовый результат": 237.5, "usd__Курс перекрытия": 3.425,
     "usd__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                         "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}},
     "eur__kurs_prinyato": 3.40, "eur__kurs_vydano": 3.45, "eur__Сумма принято": 4500, "eur__Сумма выдано": 5000,
     "eur__Финансовый результат": 237.5, "eur__Курс перекрытия": 3.425,
     "eur__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                         "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}},
     "rub__kurs_prinyato": 3.40, "rub__kurs_vydano": 3.45, "rub__Сумма принято": 4500, "rub__Сумма выдано": 5000,
     "rub__Финансовый результат": 237.5, "rub__Курс перекрытия": 3.425,
     "rub__Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                         "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},

    # {"id": 701, "kurs_prinyato": 3.40, "kurs_vydano": 3.45, "nomer_punkta": 701, "Сумма принято": 4500,
    #  "Сумма выдано": 5000, "Финансовый результат": 237.5, "Курс перекрытия": 3.425,
    #  "Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
    #                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},
    # {"id": 702, "kurs_prinyato": 3.40, "kurs_vydano": 3.45, "nomer_punkta": 702, "Сумма принято": 4500,
    #  "Сумма выдано": 5000, "Финансовый результат": 237.5, "Курс перекрытия": 3.425,
    #  "Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
    #                 "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},

]


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


def get_db():
    with Session(engine) as session:
        yield session


async def get_current_user(
        request: Request, db: Session = Depends(get_db)
):
    token = request.session.get("access_token")
    if not token:
        return None
    user = db.exec(select(User).where(User.user_name == token)).first()
    return user


async def get_current_active_user(request: Request, current_user: Annotated[Optional[User], Depends(get_current_user)]):
    if not current_user:
        request.session["redirect_url"] = str(request.url)
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
            detail="Not authenticated",
        )
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

async def get_features():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8002/features")
        response.raise_for_status()
        return response.json()

async def get_predictions(features,):
    # Simulate making predictions using an external API
    # Replace this with your actual API call
    async with httpx.AsyncClient() as client:
        logger.warning(features)
        response = await client.post("http://localhost:8002/predict", json={"features": features,})
        response.raise_for_status()
        return response.json()



@app.get("/favicon.ico")
async def favicon():
    # Option 1: Serve a favicon file from your static directory
    # return FileResponse("static/favicon.ico")

    # Option 2: Return a "No Content" response (if you don't have a favicon)
    return PlainTextResponse(status_code=204)

@app.get("/login", response_class=HTMLResponse)
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
        return RedirectResponse(url="/login?error=Incorrect username or password",
                                status_code=status.HTTP_303_SEE_OTHER)

    request.session["access_token"] = user.user_name
    redirect_url = request.session.pop("redirect_url", None)
    if redirect_url:
        return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    else:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
async def logout(request: Request):
    request.session.pop("access_token", None)
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.exec(select(User).where(User.user_name == user.user_name)).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(id=user.id, user_name=user.user_name, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}


@app.get("/", response_class=HTMLResponse)
async def list_view(
        request: Request,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
):
    request.session['last_page'] = "list.html"
    if not (current_user.user_name.lower().startswith('ca')):
        return RedirectResponse(url=f"/{current_user.id}", status_code=status.HTTP_303_SEE_OTHER)

    form_data = request.session.get("form_data")
    if form_data:
        form_data = json.loads(form_data)
        logger.critical(form_data)
    else:
        form_data = {}

    logger.debug(items)

    request.session["items"] = items
    return templates.TemplateResponse("list.html", {"request": request, "items": request.session.get("items"), "current_user": current_user,
                                                    "form_data": form_data, "header_columns":header_columns})


@app.get("/{item_id}/", response_class=HTMLResponse)
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
):
    request.session['last_page'] = "list.html"
    if item_id == 369:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    if (current_user.id != item_id) and not (current_user.user_name.lower().startswith('ca')):
        return RedirectResponse(url=f"/{current_user.id}", status_code=status.HTTP_303_SEE_OTHER)



    form_data = request.session.get("form_data")
    if form_data:
        try:
            form_data = json.loads(form_data)
            logger.debug(f"Form data in detail view for item {item_id}: {form_data}")
            kurs_prinyato_key = f"item_{item_id}_kurs_prinyato"
            kurs_vydano_key = f"item_{item_id}_kurs_vydano"
            logger.debug(f"Checking for key '{kurs_prinyato_key}': {form_data.get(kurs_prinyato_key)}")
            logger.debug(f"Checking for key '{kurs_vydano_key}': {form_data.get(kurs_vydano_key)}")
        except json.JSONDecodeError:
            form_data = {}
    else:
        form_data = {}

    detail_view_items = [item for item in items if item.get('nomer_punkta') == item_id]

    request.session["items"] = detail_view_items
    return templates.TemplateResponse("detail.html",
                                      {"request": request, "items": request.session.get("items"),
                                       "current_user": current_user, "form_data": form_data, "header_columns": header_columns})


@app.post("/calculate")
async def calculate(request: Request, current_user: Annotated[User, Depends(get_current_active_user)], mode: Literal['fast', 'slow', 'explain'] = Form(...)):
    form_data = await request.form()
    old_form_data_str = request.session.get("form_data", "{}")  # Get as string, default to empty JSON string
    logger.debug(f"Old form data (str): {old_form_data_str}, Type: {type(old_form_data_str)}")

    try:
        old_form_data = json.loads(old_form_data_str)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in old_form_data, resetting.")
        old_form_data = {}

    form_data = dict(form_data)
    logger.debug(f"New form data: {form_data}, Type: {type(form_data)}")

    # Update old_form_data with the new form_data, overwriting only existing keys
    old_form_data.update(form_data)

    request.session["form_data"] = json.dumps(old_form_data)

    # Extract form data based on request type and URL
    referer = request.headers.get("referer")
    if referer.endswith("/"):
        # Process list view form data
        print("List view form data dict:", old_form_data)  # Log the merged data
    elif referer.split("/")[-2].isdigit():
        item_id = int(referer.split("/")[-2])
        # Process details view form data
        print(f"Details view form data dict (item_id: {item_id}):", old_form_data)  # Log the merged data
    else:
        print("referer", referer)

    # get cache
    timestamp = cached_features.get('timestamp')
    if timestamp and datetime.now() - pd.Timestamp(timestamp).to_pydatetime() < datetime.timedelta(hours=2, minutes=55):
        # cache is valid
        features = cached_features.get('features')
    else:
        features = None
    # pass slow, fast, explain
    if mode == 'slow' or (not features):
        # if request.session.get('features') is not valid, generate: input_x = request.get(url='localhost:8002/features') to get data and put it into cache
        features = await get_features()
        cached_features['features'] = features
        cached_features['timestamp'] = str(datetime.now())
        logger.debug(f'slow calculation: len - {len(features)}, timestamp - {cached_features["timestamp"]}')

    predictions = await get_predictions(features)
    if mode == 'explain':
        # shap explainer
        pass
    logger.info(f'predictions - {predictions}, {request.session.get("last_page")}')

    # Redirect back to the page where the form was submitted
    return templates.TemplateResponse("list.html",
                                      context={
                                          "request": request,
                                           "items": request.session.get("items"),  # pass predictions instead
                                           "current_user": current_user,
                                           "form_data": form_data,
                                           "header_columns": header_columns
                                      })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)