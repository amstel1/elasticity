from fastapi import FastAPI, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Annotated, Optional
import uvicorn
from fastapi import Depends
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
header_columns = {"kurs_prinyato", "kurs_vydano", "nomer_punkta", "Сумма принято (3h, прогноз)", "Сумма выдано (3h, прогноз)", "Финансовый результат", "Курс перекрытия"}
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
    {"id": 700, "kurs_prinyato": 3.40, "kurs_vydano": 3.45, "nomer_punkta": 700, "Сумма принято (3h, прогноз)": 4500, "Сумма выдано (3h, прогноз)": 5000, "Финансовый результат": 237.5, "Курс перекрытия": 3.425, "Конкуренты":{"Альфа-Банк":{"Покупка": 3.41, "Продажа": 3.44}, "ВТБ-Банк":{"Покупка": 3.411, "Продажа": 3.439}}},
    {"id": 701, "kurs_prinyato": 3.40, "kurs_vydano": 3.45, "nomer_punkta": 701, "Сумма принято (3h, прогноз)": 4500,
     "Сумма выдано (3h, прогноз)": 5000, "Финансовый результат": 237.5, "Курс перекрытия": 3.425,
     "Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                    "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},
    {"id": 702, "kurs_prinyato": 3.40, "kurs_vydano": 3.45, "nomer_punkta": 702, "Сумма принято (3h, прогноз)": 4500,
     "Сумма выдано (3h, прогноз)": 5000, "Финансовый результат": 237.5, "Курс перекрытия": 3.425,
     "Конкуренты": {"Альфа-Банк": {"Покупка": 3.41, "Продажа": 3.44},
                    "ВТБ-Банк": {"Покупка": 3.411, "Продажа": 3.439}}},

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
    if not (current_user.user_name.lower().startswith('ca')):
        return RedirectResponse(url=f"/{current_user.id}", status_code=status.HTTP_303_SEE_OTHER)

    form_data = request.session.get("form_data")
    if form_data:
        form_data = json.loads(form_data)
        logger.critical(form_data)
    else:
        form_data = {}

    logger.debug(items)
    return templates.TemplateResponse("list.html", {"request": request, "items": items, "current_user": current_user,
                                                    "form_data": form_data, "header_columns":header_columns})


@app.get("/{item_id}/", response_class=HTMLResponse)
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
):
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
    return templates.TemplateResponse("detail.html",
                                      {"request": request, "items": detail_view_items,
                                       "current_user": current_user, "form_data": form_data, "header_columns": header_columns})


@app.post("/calculate")
async def calculate(request: Request, current_user: Annotated[User, Depends(get_current_active_user)]):
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

    # Redirect back to the page where the form was submitted
    return RedirectResponse(url=referer, status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)