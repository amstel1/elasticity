

import numpy as np
import requests
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
from starlette.middleware.sessions import SessionMiddleware
import json
from loguru import logger
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import pickle
import sys

logger.remove()
logger.add(sys.stderr, format="{time} | {level} | {module}.{function}:{line} - {message}", level="INFO")
logger.add("app.log", rotation="100 MB", format="{time} | {level} | {module}.{function}:{line} - {message} {exception}", level="DEBUG")

SESSION_FILES_DIR = Path("./session_data")
SESSION_FILES_DIR.mkdir(exist_ok=True)
DATABASE_URL = "sqlite:///./elasticity_model.db"
SECRET_KEY = "your_secret_key"
GET_DATA_API_URL = "http://localhost:8002/get_data"
GET_FEATURES_API_URL = "http://localhost:8002/get_features"
PREDICT_API_URL = "http://localhost:8002/predict"
MAKE_PROPOSED_KURSES_API_URL = "http://localhost:8002/make_proposed_kurses"
LOGIN_ERROR_URL = "/login?error=Incorrect username or password"
LOGIN_URL = "/login"
ROOT_URL = "/"
HEADER_COLUMNS = (
    "nomer_punkta",
    "usd__kurs_prinyato", "usd__Сумма принято", "usd__Курс перекрытия", "usd__Сумма выдано", "usd__kurs_vydano", "usd__Финансовый результат",
    "eur__kurs_prinyato", "eur__Сумма принято", "eur__Курс перекрытия", "eur__Сумма выдано", "eur__kurs_vydano", "eur__Финансовый результат",
    "rub__kurs_prinyato", "rub__Сумма принято", "rub__Курс перекрытия", "rub__Сумма выдано", "rub__kurs_vydano", "rub__Финансовый результат",
)
cached_features: Dict = {}
engine = create_engine(DATABASE_URL, echo=True)
PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="/token")

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY, session_cookie="session", max_age=3600, https_only=False,) # Explicitly using session_cookie, should trigger FileSessionBackend
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return PASSWORD_CONTEXT.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_db():
    with Session(engine) as session:
        yield session

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

class FormRates(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_name: str = Field(unique=False, index=True)
    rates_from_form : str = Field()
    flag_make_proposed_kurses: bool = True
    inserted_datetime: datetime = Field(default_factory=lambda: pytz.timezone('Europe/Minsk').localize(datetime.now()))
    mode: str = Field()


class Prediction(SQLModel, table=True, extend_existing=True):
    model_config = {'extend_existing': True}
    row_id: int | None = Field(default=None, primary_key=True)
    inserted_datetime: datetime = Field(default_factory=lambda: pytz.timezone('Europe/Minsk').localize(datetime.now()))
    now: str
    user_name: str
    nomer_punkta: int
    usd__kurs_prinyato: float
    usd__kurs_vydano: float
    usd__Сумма_принято: float
    usd__Сумма_выдано: float
    usd__Финансовый_результат: float
    usd__Курс_перекрытия: float
    eur__kurs_prinyato: float
    eur__kurs_vydano: float
    eur__Сумма_принято: float
    eur__Сумма_выдано: float
    eur__Финансовый_результат: float
    eur__Курс_перекрытия: float
    rub__kurs_prinyato: float
    rub__kurs_vydano: float
    rub__Сумма_принято: float
    rub__Сумма_выдано: float
    rub__Финансовый_результат: float
    rub__Курс_перекрытия: float


def insert_predictions_to_db(
        data: list[Dict],
        now: str,
        user_name: str,
):
    df = pd.DataFrame(data)
    df.columns = [x.replace(' ', '_') for x in df.columns]
    df['now'] = now
    df['user_name'] = user_name
    predictions = [Prediction(**prediction_kwargs) for prediction_kwargs in df.to_dict(orient='records')]
    with Session(engine) as session:
        try:
            session.add_all(predictions)
            session.commit()
        except Exception as e:
            logger.exception(e)
            logger.error(f'error insert_predictions_to_db: {e}')
            session.rollback()
            raise HTTPException(status_code=500, detail=f"error insert_predictions_to_db: {e}")

def insert_form_rates( user_name: str, rates_from_form: Dict, flag_make_proposed_kurses: bool, mode: Literal['fast', 'slow']):
    with Session(engine) as session:
        try:
            form_rates = FormRates(
                mode=mode,
                user_name=user_name,
                rates_from_form=rates_from_form,
                flag_make_proposed_kurses=flag_make_proposed_kurses,
                inserted_datetime = datetime.now()
            )
            session.add(form_rates)
            session.commit()
            session.refresh(form_rates)
        except Exception as e:
            logger.exception(e)
            logger.error(f'error saving session data (insert_form_rates): {e}')
            session.rollback()
            raise HTTPException(status_code=500, detail=f"error saving session data (insert_form_rates): {e}")


def select_form_rates(request: Request):
    user_name = request.session.get("access_token")
    if not user_name:
        return {}
    with Session(engine) as session:
        try:
            form_rates = session.exec(select(FormRates).where(FormRates.user_name == user_name).order_by(FormRates.inserted_datetime.desc())).first()
            if form_rates:
                return form_rates
            else:
                return {}
        except Exception as e:
            logger.error(f"select_form_rates - error: {e}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"select_form_rates - error: {e}")


@app.on_event("startup")
def on_startup():
    global IS_DEBUG_FLAG
    IS_DEBUG_FLAG = True
    # global now
    now = '2024-09-17 15:45:28'
    create_db_and_tables()


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

@logger.catch(reraise=False)
def prepare_4_insertion(data_to_render: List[Dict]):
    results = {}
    for item in data_to_render:
        pos = item.get('nomer_punkta')
        for currency in ('usd', 'eur', 'rub'):
            for operation_type in ('prinyato', 'vydano'):
                results[f'{pos}__{currency}__kurs_{operation_type}'] = item.get(f'{currency}__kurs_{operation_type}')
    return results

@logger.catch(reraise=False)
def handle_view(db_response, form_data: dict):
    if db_response and (datetime.now() - db_response.inserted_datetime) < np.timedelta64(2, 'h'):
        rates_from_session = json.loads(db_response.rates_from_form)
        rates_from_session = {k: float(v) for k, v in rates_from_session.items() if '__' in k}
        flag_make_proposed_kurses = db_response.flag_make_proposed_kurses
        mode = db_response.mode
    else:
        rates_from_session = {}
        flag_make_proposed_kurses = True
        mode = 'fast'

    if form_data or rates_from_session:
        rates_from_session.update(form_data)
        try:
            len(mode)
        except Exception as e:
            mode = 'fast'
    return rates_from_session, flag_make_proposed_kurses, mode

@logger.catch(reraise=False)
def process_before_viewing(rate_from_form={}, mode: Literal['fast', 'slow'] = 'fast', flag_make_proposed_kurses: bool = True, now: str = str(datetime.now())):
    '''returns: data_to_render (ready for template) '''
    features = fetch_features(mode=mode, kind='baseline_kurs')
    baseline_kurs_data = features.get('baseline_kurs_data')
    baseline_kurs_calc_datetime = features.get('baseline_kurs_calc_datetime')

    features = fetch_features(mode=mode, kind='myfin')
    myfin_data = features.get('myfin_data')
    myfin_calc_datetime = features.get('myfin_calc_datetime')

    try:
        baseline_kurs_data
    except Exception as e:
        features = fetch_features(mode=mode, kind='baseline_kurs')
        baseline_kurs_data = features.get('baseline_kurs_data')
        baseline_kurs_calc_datetime = features.get('baseline_kurs_calc_datetime')

    try:
        myfin_data
    except Exception as e:
        features = fetch_features(mode=mode, kind='myfin')
        myfin_data = features.get('myfin_data')
        myfin_calc_datetime = features.get('myfin_calc_datetime')


    # init load
    features = None
    if not IS_DEBUG_FLAG:
        features = fetch_features(mode=mode, kind='voo')  # three kinds
    if IS_DEBUG_FLAG:
        # features = fetch_features(mode=mode, kind='voo')  # three kinds
        # with open('debug_features.pkl', 'wb') as f:
        #     pickle.dump(features, f)
        with open('debug_features.pkl', 'rb') as f:
            features = pickle.load(f)
    assert isinstance(cached_features, dict)

    voo_data = features.get('voo_data')
    voo_calc_datetime = features.get('voo_calc_datetime')

    if not flag_make_proposed_kurses:
        # use kurs from the forms
        # to: get the kurses from the user's input
        assert rate_from_form
        prefinal_data = fetch_data_to_render(
            now=now,
            flag_make_proposed_kurses=flag_make_proposed_kurses,  # edit24: if - else depends on the
            voo_data=voo_data,
            voo_calc_datetime=voo_calc_datetime,
            baseline_kurs_data=baseline_kurs_data,
            baseline_kurs_calc_datetime=baseline_kurs_calc_datetime,
            myfin_data=myfin_data,
            myfin_calc_datetime=myfin_calc_datetime,
            rate_from_form=rate_from_form,
        )
    else:
        # model must propose kurses
        prefinal_data = fetch_data_to_render(
            now=now,
            flag_make_proposed_kurses=True,
            voo_data=voo_data,
            voo_calc_datetime=voo_calc_datetime,
            baseline_kurs_data=baseline_kurs_data,
            baseline_kurs_calc_datetime=baseline_kurs_calc_datetime,
            myfin_data=myfin_data,
            myfin_calc_datetime=myfin_calc_datetime,
        )

    usd_profit_df = pd.DataFrame(prefinal_data.get('data', {}).get('usd'))
    usd_profit_df = usd_profit_df.rename(columns={
        'nomer punkta': 'nomer_punkta',
        'proposed kurs prinyato': 'usd__kurs_prinyato',
        'proposed kurs vydano': 'usd__kurs_vydano',
        'predicted volume prinyato': 'usd__Сумма принято',
        'predicted volume vydano': 'usd__Сумма выдано',
        'profit_saldo': 'usd__Финансовый результат',
        'baseline kurs prinyato': 'usd__Курс перекрытия',
    })
    usd_profit_df['nomer_punkta'] = usd_profit_df['nomer_punkta'].astype(int)
    usd_profit_df['id'] = usd_profit_df['nomer_punkta']
    usd_profit_df = usd_profit_df[['nomer_punkta', 'usd__kurs_prinyato', 'usd__kurs_vydano', 'usd__Сумма принято', 'usd__Сумма выдано', 'usd__Финансовый результат', 'usd__Курс перекрытия']]

    eur_profit_df = pd.DataFrame(prefinal_data.get('data', {}).get('eur'))
    eur_profit_df = eur_profit_df.rename(columns={
        'nomer punkta': 'nomer_punkta',
        'proposed kurs prinyato': 'eur__kurs_prinyato',
        'proposed kurs vydano': 'eur__kurs_vydano',
        'predicted volume prinyato': 'eur__Сумма принято',
        'predicted volume vydano': 'eur__Сумма выдано',
        'profit_saldo': 'eur__Финансовый результат',
        'baseline kurs prinyato': 'eur__Курс перекрытия',
    })
    eur_profit_df['nomer_punkta'] = eur_profit_df['nomer_punkta'].astype(int)
    eur_profit_df['id'] = eur_profit_df['nomer_punkta']
    eur_profit_df = eur_profit_df[['nomer_punkta', 'eur__kurs_prinyato', 'eur__kurs_vydano', 'eur__Сумма принято', 'eur__Сумма выдано', 'eur__Финансовый результат', 'eur__Курс перекрытия']]

    rub_profit_df = pd.DataFrame(prefinal_data.get('data', {}).get('rub'))
    rub_profit_df = rub_profit_df.rename(columns={
        'nomer punkta': 'nomer_punkta',
        'proposed kurs prinyato': 'rub__kurs_prinyato',
        'proposed kurs vydano': 'rub__kurs_vydano',
        'predicted volume prinyato': 'rub__Сумма принято',
        'predicted volume vydano': 'rub__Сумма выдано',
        'profit_saldo': 'rub__Финансовый результат',
        'baseline kurs prinyato': 'rub__Курс перекрытия',
    })
    rub_profit_df['nomer_punkta'] = rub_profit_df['nomer_punkta'].astype(int)
    rub_profit_df['id'] = rub_profit_df['nomer_punkta']
    rub_profit_df = rub_profit_df[['nomer_punkta', 'rub__kurs_prinyato', 'rub__kurs_vydano', 'rub__Сумма принято', 'rub__Сумма выдано', 'rub__Финансовый результат', 'rub__Курс перекрытия']]

    df = usd_profit_df.merge(eur_profit_df, 'left', on=('nomer_punkta',)).merge(rub_profit_df, 'left', on=('nomer_punkta',))
    df.sort_values('nomer_punkta', inplace=True)
    data_to_render = df.to_dict(orient='records')
    datetimes = {
        'voo_calc_datetime':voo_calc_datetime,
        'baseline_kurs_calc_datetime':baseline_kurs_calc_datetime,
        'myfin_calc_datetime':myfin_calc_datetime,
    }
    return myfin_data, data_to_render, datetimes


@logger.catch(reraise=False)
def fetch_features( mode: Literal['slow', 'fast'], kind=Literal['voo', 'baseline_kurs', 'myfin']) -> Dict:
    """Fetches features"""
    # use form_data to get user input
    data_to_fetch = {}  # same struct as cached features
    assert isinstance(cached_features, Dict)

    if kind == 'voo':
        if cached_features.get('voo_calc_datetime') and datetime.now() - pd.Timestamp(cached_features.get('voo_calc_datetime')).to_pydatetime() < timedelta(hours=2, minutes=55):
            if cached_features.get('voo_data') and mode != 'slow':
                data_to_fetch['voo_data'] = cached_features.get('voo_data')
        else:
            # data is missing, expired or mode=slow
            response = requests.post(url=GET_FEATURES_API_URL, json={'kind': 'voo'})  # response may be of three kinds
            response.raise_for_status()
            resp = response.json()
            data_to_fetch['voo_data'] = resp.get('data')
            data_to_fetch['voo_calc_datetime'] = resp.get('calc_datetime')
            cached_features['voo_data'] = resp.get('data')
            cached_features['voo_calc_datetime'] = resp.get('calc_datetime')
    elif kind == 'baseline_kurs':
        response = requests.post(url=GET_FEATURES_API_URL, json={'kind': 'baseline_kurs'})  # response may be of three kinds
        response.raise_for_status()
        resp = response.json()
        data_to_fetch['baseline_kurs_data'] = resp.get('data')
        data_to_fetch['baseline_kurs_calc_datetime'] = resp.get('calc_datetime')
        cached_features['baseline_kurs_data'] = resp.get('data')
        cached_features['baseline_kurs_calc_datetime'] = resp.get('calc_datetime')
    elif kind == 'myfin':
        response = requests.post(url=GET_FEATURES_API_URL, json={'kind': 'myfin'})  # response may be of three kinds
        response.raise_for_status()
        resp = response.json()
        data_to_fetch['myfin_data'] = resp.get('data')
        data_to_fetch['myfin_calc_datetime'] = resp.get('calc_datetime')
        cached_features['myfin_data'] = resp.get('data')
        cached_features['myfin_calc_datetime'] = resp.get('calc_datetime')
    return data_to_fetch

@logger.catch(reraise=False)
def fetch_data_to_render(
        now: str = str(datetime.now()),
        flag_make_proposed_kurses: bool = False,
        voo_data=None,
        voo_calc_datetime=None,
        baseline_kurs_data=None,
        baseline_kurs_calc_datetime=None,
        myfin_data=None,
        myfin_calc_datetime=None,
        rate_from_form: Dict = {},
) -> List[Dict]:

    if flag_make_proposed_kurses:
        baseline_kurs_data_df = pd.DataFrame(baseline_kurs_data)
        if IS_DEBUG_FLAG:
            baseline_kurs_data_df['this_currency'] = baseline_kurs_data_df['Валюта']
            baseline_kurs_data_df['c_rate'] = baseline_kurs_data_df['Курс продажи']  # okay for debug
        baseline_kurs_dict = baseline_kurs_data_df.set_index('this_currency')['c_rate'].to_dict()
        resp = requests.post(MAKE_PROPOSED_KURSES_API_URL, json=baseline_kurs_dict)
        proposed_kurses = resp.json()
    else:
        # user overridden
        baseline_kurs_data_df = pd.DataFrame(baseline_kurs_data)
        if IS_DEBUG_FLAG:
            baseline_kurs_data_df['this_currency'] = baseline_kurs_data_df['Валюта']
            baseline_kurs_data_df['c_rate'] = baseline_kurs_data_df['Курс продажи']  # okay for debug
        baseline_kurs_dict = baseline_kurs_data_df.set_index('this_currency')['c_rate'].to_dict()
        assert isinstance(rate_from_form, Dict)
        pos_curr_rate = rate_from_form

    if flag_make_proposed_kurses:
        resp = requests.post(PREDICT_API_URL, json={'voo_data': voo_data, 'proposed_kurses': proposed_kurses, 'baseline_kurs_dict': baseline_kurs_dict, 'now': now})  # profits_df
        final_data = resp.json()
    else:
        resp = requests.post(PREDICT_API_URL,
                             json={'voo_data': voo_data, 'pos_curr_rate': pos_curr_rate, 'baseline_kurs_dict':baseline_kurs_dict, 'now': now})  # todo: add baseline_kurs, returns: profits_df
        final_data = resp.json()

    return final_data


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse('static/sber.ico')

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

@logger.catch(reraise=False)
@app.get(ROOT_URL, response_class=HTMLResponse, name="list_view")
async def list_view(
        request: Request,
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    logger.info(f'now: {now}')
    logger.info(f'IS_DEBUG_FLAG: {IS_DEBUG_FLAG}')
    user_name = request.session.get('access_token')
    db_response = select_form_rates(request)
    form_data = dict(await request.form())
    if not IS_DEBUG_FLAG:
        now = str(datetime.now())

    rates_from_session, flag_make_proposed_kurses, mode = handle_view(db_response, form_data)

    _, data_to_render, datetimes = process_before_viewing(
        rate_from_form=rates_from_session,
        mode=mode,
        flag_make_proposed_kurses=flag_make_proposed_kurses,
        now=now,
    )
    # save predictions
    if user_name == 'airflow':
        insert_predictions_to_db(data=data_to_render, now=now, user_name=user_name)

    if current_user.id != 369:
        data_to_render = [item for item in data_to_render if int(item.get('nomer_punkta', 99999) // 100) == current_user.id]

    rates_from_form = prepare_4_insertion(data_to_render)
    # save session
    insert_form_rates(user_name=user_name,
                      rates_from_form=json.dumps(rates_from_form),
                      flag_make_proposed_kurses=flag_make_proposed_kurses,
                      mode=mode,
                      )
    return templates.TemplateResponse("list.html", {"request": request, "items": data_to_render, "current_user": current_user,
                                                     "header_columns": HEADER_COLUMNS, "datetimes": datetimes})


@app.get("/{item_id}/", response_class=HTMLResponse, name="detail_view")
@logger.catch(reraise=False)
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    user_name = request.session.get('access_token')
    db_response = select_form_rates(request)
    form_data = dict(await request.form())
    now = str(datetime.now())

    rates_from_session, flag_make_proposed_kurses, mode = handle_view(db_response, form_data)

    myfin_data, data_to_render, datetimes = process_before_viewing(
        rate_from_form=rates_from_session,
        mode=mode,
        flag_make_proposed_kurses=flag_make_proposed_kurses,
        now=now,
    )
    # save predictions
    if user_name == 'airflow':
        insert_predictions_to_db(data=data_to_render, now=now, user_name=user_name)

    detail_view_data_to_render = [x for x in data_to_render if x.get('nomer_punkta') == item_id]
    detail_view_competitor = {}

    for currency in ('usd', 'eur', 'rub'):
        this_currency_list = myfin_data.get(currency)
        currency_this_point = [x for x in this_currency_list if x.get('nomer_punkta') == item_id]  # todo: check if i need sorting
        temp = pd.DataFrame(currency_this_point)
        temp['rank_min'] = temp.groupby('operation_type')['value'].transform('rank', ascending=False, method='dense')
        temp['rank_max'] = temp.groupby('operation_type')['value'].transform('rank', ascending=True, method='dense')
        currency_this_point_retval = temp.to_dict(orient='records')
        detail_view_competitor[currency] = currency_this_point_retval


    rates_from_form = prepare_4_insertion(data_to_render)
    # save session
    insert_form_rates(user_name=user_name,
                      rates_from_form=json.dumps(rates_from_form),
                      flag_make_proposed_kurses=flag_make_proposed_kurses,
                      mode=mode,
                      )
    return templates.TemplateResponse("detail.html",
                                      {"request": request, "items": detail_view_data_to_render, "competitors": detail_view_competitor,
                                       "current_user": current_user, "header_columns": HEADER_COLUMNS, "datetimes":datetimes})


@logger.catch(reraise=False)
def validate_form_data(form_data: Dict) -> Dict[str, float]:
    try:
        validate_form_data = {}
        for k, v in form_data.items():
            if k.startswith('item_'):
                validate_form_data[k] = float(v)
            else:
                validate_form_data[k] = v
        return validate_form_data
    except Exception as e:
        logger.error(f"validate_form_data - error: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"validate_form_data - error: {e}")


@app.post("/calculate", name="calculate")
@logger.catch(reraise=False)
async def calculate(
    request: Request,
    current_user: Annotated[User, Depends(get_current_active_user)],
    mode: Literal['fast', 'slow',] = Form(...)
):
    user_name = request.session.get('access_token')
    db_response = select_form_rates(request)
    form_data = dict(await request.form())
    form_data = validate_form_data(form_data)

    if mode == 'fast':
        if db_response and (datetime.now() - db_response.inserted_datetime) < np.timedelta64(3, 'h'):
            rates_from_session = json.loads(db_response.rates_from_form)
            rates_from_session = {k: float(v) for k,v in rates_from_session.items() if '__' in k}
        else:
            rates_from_session = {}

        if not rates_from_session:
            # первый заход или сессия устарела -> берем из формы
            flag_make_proposed_kurses = True
            insert_form_rates(user_name=user_name,
                              rates_from_form=json.dumps(form_data),
                              flag_make_proposed_kurses=flag_make_proposed_kurses,
                              mode=mode)
        else:
            # saved_rates exist
            # compare old and new
            if rates_from_session == form_data:
                # равносильно обновлению страницы
                # user did not change anything, IGNORE kurses from the form
                flag_make_proposed_kurses = True
            else:
                # user did change something, TAKE kurses from the form
                flag_make_proposed_kurses = False
                rates_from_session.update(form_data)
                insert_form_rates(user_name=user_name,
                                  rates_from_form=json.dumps(rates_from_session),
                                  flag_make_proposed_kurses=flag_make_proposed_kurses,
                                  mode=mode)
    elif mode == 'slow':
        # модель перезаписывает курсы пользователя
        flag_make_proposed_kurses = True
        insert_form_rates(user_name=user_name,
                          rates_from_form=json.dumps({}),
                          flag_make_proposed_kurses=flag_make_proposed_kurses,
                          mode=mode)


    referer = request.headers.get("referer")
    if not referer:
        referer = ROOT_URL
    return RedirectResponse(url=referer, status_code=status.HTTP_303_SEE_OTHER)


# --- Main Entry Point ---
if __name__ == "__main__":
    IS_DEBUG_FLAG = None
    now = None
    uvicorn.run(app, host="0.0.0.0", port=8000,)
