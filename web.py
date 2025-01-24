IS_DEBUG_FLAG = True

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
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware

import json
from loguru import logger
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import time
import pickle
import sys  # Import sys for getsizeof

# --- Constants and Configuration ---
SESSION_FILES_DIR = Path("./session_data")
SESSION_FILES_DIR.mkdir(exist_ok=True)
DATABASE_URL = "sqlite:///./elasticity_model.db"
SECRET_KEY = "your_secret_key" # Replace with a strong, randomly generated secret key in production
GET_DATA_API_URL = "http://localhost:8002/get_data"
GET_FEATURES_API_URL = "http://localhost:8002/get_features"
PREDICT_API_URL = "http://localhost:8002/predict"
EXPLAIN_API_URL = "http://localhost:8002/explain"
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


# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY, session_cookie="session", max_age=3600, https_only=False,) # Explicitly using session_cookie, should trigger FileSessionBackend
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Event Handlers ---
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/set-session")
async def set_session(request: Request):
    request.session['my_data'] = "Hello from session!"
    return {"message": 'Session data set'}

@app.get("/get-session")
async def get_session(request: Request):
    session_data = request.session.get('my_data')
    logger.critical(session_data)
    return {"session_data": session_data}

# --- Dependency Functions ---
async def get_current_user(
        request: Request, db: Session = Depends(get_db)
) -> Optional[User]:
    """Retrieves the current user based on the session token."""
    user_name = request.session.get("access_token")
    if not user_name:
        return None
    user = db.exec(select(User).where(User.user_name == user.user_name)).first()
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


def process_before_viewing(rate_from_form={}, mode: Literal['fast', 'slow'] = 'fast', flag_make_proposed_kurses: bool = True):
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
        features = fetch_features(mode=mode, kind='baseline_kurs')  # todo: three kinds
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
    features = fetch_features(mode=mode, kind='voo')  # three kinds
    logger.debug('features = calculated')
    assert isinstance(cached_features, dict)
    logger.debug(f'len cached_features: {len(cached_features)}')

    # logger.debug(list(f'key: {k}, len v: {len(v)}' for k,v in cached_features.items()))
    voo_data = features.get('voo_data')
    voo_calc_datetime = features.get('voo_calc_datetime')

    if not flag_make_proposed_kurses:
        # use kurs from the forms
        # to: get the kurses from the user's input
        assert rate_from_form
        prefinal_data = fetch_data_to_render(
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

    df = usd_profit_df.merge(eur_profit_df, 'left', on=('nomer_punkta',)).merge(rub_profit_df, 'left', on=('nomer_punkta', ))
    df.sort_values('nomer_punkta', inplace=True)
    data_to_render = df.to_dict(orient='records')
    logger.info(f'data to render 1: {type(data_to_render)}')
    return myfin_data, data_to_render


# --- API Interaction Functions ---
def fetch_features( mode: Literal['slow', 'fast'], kind=Literal['voo', 'baseline_kurs', 'myfin']) -> Dict:
    """Fetches features from the external API, using a cache."""
    # use form_data to get user input
    logger.info('in fetch_features')
    data_to_fetch = {}  # same struct as cached features

    # check to three kinds of cache separately
    assert isinstance(cached_features, Dict)
    # cached_features['voo_data']
    # cached_features['voo_calc_datetime']

    # cached_features['baseline_kurs_data']
    # cached_features['baseline_kurs_calc_datetime']

    # cached_features['myfin_data']
    # cached_features['myfin_calc_datetime']
    # logger.debug(f'cached_features - {cached_features}')
    if kind == 'voo':
        if cached_features.get('voo_calc_datetime') and datetime.now() - pd.Timestamp(cached_features.get('voo_calc_datetime')).to_pydatetime() < timedelta(hours=2, minutes=55):
            logger.debug(f'1')
            if cached_features.get('voo_data') and mode != 'slow':
                logger.debug(f'2')
                data_to_fetch['voo_data'] = cached_features.get('voo_data')
        else:
            # data is missing, expired or mode=slow
            logger.info(f'before - {GET_FEATURES_API_URL}')
            response = requests.post(url=GET_FEATURES_API_URL, json={'kind': 'voo'})  # response may be of three kinds
            logger.info(f'after - {GET_FEATURES_API_URL}')
            response.raise_for_status()
            resp = response.json()
            data_to_fetch['voo_data'] = resp.get('data')
            cached_features['voo_data'] = resp.get('data')
            cached_features['voo_calc_datetime'] = resp.get('calc_datetime')
    elif kind == 'baseline_kurs':
        logger.debug(f'3')
        response = requests.post(url=GET_FEATURES_API_URL, json={'kind': 'baseline_kurs'})  # response may be of three kinds
        response.raise_for_status()
        resp = response.json()
        data_to_fetch['baseline_kurs_data'] = resp.get('data')
        cached_features['baseline_kurs_data'] = resp.get('data')
        cached_features['baseline_kurs_calc_datetime'] = resp.get('calc_datetime')
    elif kind == 'myfin':
        response = requests.post(url=GET_FEATURES_API_URL, json={'kind': 'myfin'})  # response may be of three kinds
        response.raise_for_status()
        resp = response.json()
        data_to_fetch['myfin_data'] = resp.get('data')
        cached_features['myfin_data'] = resp.get('data')
        cached_features['myfin_calc_datetime'] = resp.get('calc_datetime')
        logger.debug(f'4')
    return data_to_fetch


def fetch_data_to_render(
        flag_make_proposed_kurses: bool = False,
        voo_data=None,
        voo_calc_datetime=None,
        baseline_kurs_data=None,
        baseline_kurs_calc_datetime=None,
        myfin_data=None,
        myfin_calc_datetime=None,
        rate_from_form: Dict = {},
) -> List[Dict]:
    '''
    # todo: explain

    2. /predict
    вызвать make_predictions - 3 валюты * 2 типа операций = 6 раз

    3. /optimize_profit
    для каждой из 3 валют
    вызвать optimize_profit
    '''

    # для одной пары - ?

    if flag_make_proposed_kurses:
        baseline_kurs_data_df = pd.DataFrame(baseline_kurs_data)
        logger.info(f'baseline_kurs_data.head(2) - {baseline_kurs_data_df.head(2)}')
        logger.info(f'baseline_kurs_data.shape - {baseline_kurs_data_df.shape}')
        if IS_DEBUG_FLAG:
            baseline_kurs_data_df['this_currency'] = baseline_kurs_data_df['Валюта']
            baseline_kurs_data_df['c_rate'] = baseline_kurs_data_df['Курс продажи']  # okay for debug
        baseline_kurs_dict = baseline_kurs_data_df.set_index('this_currency')['c_rate'].to_dict()
        resp = requests.post(MAKE_PROPOSED_KURSES_API_URL, json=baseline_kurs_dict)
        proposed_kurses = resp.json()
        # logger.debug(f'proposed_kurses: {proposed_kurses}')
    else:
        baseline_kurs_data_df = pd.DataFrame(baseline_kurs_data)
        baseline_kurs_dict = baseline_kurs_data_df.set_index('this_currency')['c_rate'].to_dict()
        assert isinstance(rate_from_form, Dict)
        assert rate_from_form > 0
        logger.debug(f'rates_from_form - {rate_from_form}')
        # todo: parse from dict? rate_from_form
        # loop by nomer_punkta
        proposed_kurses = {
            'prinyato_usd_': -0.01,
            'vydano_usd': -0.01,
        } # list of floats


    if flag_make_proposed_kurses:
        logger.debug(f'voo_data - dtype: {type(voo_data)}')
        logger.debug(f'proposed_kurses - dtype: {type(proposed_kurses)}')
        resp = requests.post(PREDICT_API_URL, json={'voo_data': voo_data, 'proposed_kurses': proposed_kurses, 'baseline_kurs_dict': baseline_kurs_dict})  # profits_df
        final_data = resp.json()
        logger.info('5')
    else:
        # loop by nomer_punkta
        # container = []
        # take unique nomer_punkta from data
        # for nomer_punkta in [700,701,702,703,704,706,709,711,777]:
        #     logger.info(f'voo_data - dtype: {type(voo_data)}')
        #     logger.info(f'baseline_kurs_dict - dtype: {type(baseline_kurs_dict)}')
        #     logger.info(f'proposed_kurses - dtype: {type(proposed_kurses)}')
        resp = requests.post(PREDICT_API_URL,
                             # select only for this nomer_punkta
                             json={'voo_data': voo_data, 'proposed_kurses': proposed_kurses, 'baseline_kurs_dict':baseline_kurs_dict})  # todo: add baseline_kurs, returns: profits_df
        # container.append(resp.json())
        # final_data = pd.concat(container)
        final_data = resp.json()

    return final_data


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

    # form_data = request.session.get("form_data", None)
    # logger.info(f'initial form data -- {len(form_data)} -- {type(form_data)}')
    # if isinstance(form_data, str):
    #     form_data = json.loads(form_data)
    form_data = dict(await request.form())
    if form_data:
        form_data = validate_form_data(form_data)
        logger.info(f'calc form data -- {form_data}-- {len(form_data)} -- {type(form_data)}')
        form_data_ids = [x.get('nomer_punkta') for x in json.loads(form_data['source_id'].replace("\'", "\""))]
        current_rates_from_form = {}
        for id in form_data_ids:
            current_rates_from_form[id] = {}
            for name in ['item__usd_kurs_prinyato', 'item__usd_kurs_vydano', 'item__eur_kurs_prinyato',
                         'item__eur_kurs_vydano', 'item__rub_kurs_prinyato', 'item__rub_kurs_vydano']:
                current_rates_from_form[id][name] = form_data[name]
        request.session['rates_from_form'] = current_rates_from_form
        logger.debug(f"Session 'rates_from_form' set in list_view. Size: {sys.getsizeof(request.session['rates_from_form'])}") # ADDED LOGGING

    try:
        rates_from_form = request.session.get("rates_from_form", {})
        logger.debug(f"Session 'rates_from_form' retrieved in list_view. Size: {sys.getsizeof(rates_from_form)}") # ADDED LOGGING
    except:
        rates_from_form = {}
    hello = request.session.get("hello")
    logger.debug(f'hello - 0: {hello}')
    _, data_to_render = process_before_viewing(rates_from_form, mode='fast')
    if not rates_from_form:
        for item in data_to_render:
            id = item.get('nomer_punkta')
            rates_from_form[id] = {}
            for name in ['item__usd_kurs_prinyato', 'item__usd_kurs_vydano', 'item__eur_kurs_prinyato',
                         'item__eur_kurs_vydano', 'item__rub_kurs_prinyato', 'item__rub_kurs_vydano']:
                item_key = name.replace('item__', '').replace('usd_', 'usd__').replace('eur_', 'eur__').replace('rub_', 'rub__')
                rates_from_form[id][name] = item.get(item_key)
        request.session["rates_from_form"] = rates_from_form
        logger.debug(f"Session 'rates_from_form' (default) set in list_view. Size: {sys.getsizeof(request.session['rates_from_form'])}") # ADDED LOGGING
    logger.info(f'rates_from_form, list view: {len(rates_from_form)}, {type(rates_from_form)}, {rates_from_form}')

    if current_user.id != 369:
        data_to_render = [item for item in data_to_render if int(item.get('nomer_punkta', 99999) // 100) == current_user.id]
    request.session['hello'] = 'hello from list_view'
    return templates.TemplateResponse("list.html", {"request": request, "items": data_to_render, "current_user": current_user,
                                                     "header_columns": HEADER_COLUMNS})


@app.get("/{item_id}/", response_class=HTMLResponse, name="detail_view")
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    # form_data = request.session.get("form_data", None)
    # if isinstance(form_data, str):
    #     try:
    #         form_data = json.loads(form_data)
    #     except json.JSONDecodeError:
    #         form_data = {}

    form_data = dict(await request.form())
    if form_data:
        form_data = validate_form_data(form_data)
        logger.info(f'calc form data -- {form_data}-- {len(form_data)} -- {type(form_data)}')
        form_data_ids = [x.get('nomer_punkta') for x in json.loads(form_data['source_id'].replace("\'", "\""))]
        current_rates_from_form = {}
        for id in form_data_ids:
            current_rates_from_form[id] = {}
            for name in ['item__usd_kurs_prinyato', 'item__usd_kurs_vydano', 'item__eur_kurs_prinyato',
                         'item__eur_kurs_vydano', 'item__rub_kurs_prinyato', 'item__rub_kurs_vydano']:
                current_rates_from_form[id][name] = form_data[name]
        request.session['rates_from_form'] = current_rates_from_form
        logger.debug(f"Session 'rates_from_form' set in detail_view. Size: {sys.getsizeof(request.session['rates_from_form'])}") # ADDED LOGGING

    hello = request.session.get("hello")
    logger.debug(f'hello - 1: {hello}')
    rates_from_form = request.session.get("rates_from_form", {})
    logger.debug(f"Session 'rates_from_form' retrieved in detail_view. Size: {sys.getsizeof(rates_from_form)}") # ADDED LOGGING
    myfin_data, data_to_render = process_before_viewing(rates_from_form, mode='fast')


    logger.info(f'rates_from_form, detail view: {len(rates_from_form)}, {type(rates_from_form)}, {rates_from_form}')
    detail_view_data_to_render = [x for x in data_to_render if x.get('nomer_punkta') == item_id]

    detail_view_competitor = {}
    this_point = [x for x in myfin_data if x.get('nomer_punkta') == item_id]
    for currency in ('usd', 'eur', 'rub'):
        currency_this_point = sorted([x for x in this_point if x.get('currency') == currency], key=lambda x: (x.get('bank_name'), x.get('ranking')))
        detail_view_competitor[currency] = currency_this_point

    a = 1
    request.session['hello'] = 'hello from detail_view'
    return templates.TemplateResponse("detail.html",
                                      {"request": request, "items": detail_view_data_to_render, "competitors": detail_view_competitor,
                                       "current_user": current_user, "header_columns": HEADER_COLUMNS})


def validate_form_data(form_data: Dict) -> Dict[str, float]:
    validate_form_data = {}
    for k, v in form_data.items():
        if k.startswith('item_'):
            validate_form_data[k] = float(v)
        else:
            validate_form_data[k] = v
    return validate_form_data

@app.post("/calculate", name="calculate")
async def calculate(
    request: Request,
    current_user: Annotated[User, Depends(get_current_active_user)],
    mode: Literal['fast', 'slow',] = Form(...)
):
    logger.debug(mode)
    hello = request.session.get("hello")
    logger.debug(f'hello - 2: {hello}')
    rates_from_form = request.session.get("rates_from_form", {})
    logger.debug(f"Session 'rates_from_form' retrieved in calculate. Size: {sys.getsizeof(rates_from_form)}") # ADDED LOGGING
    form_data = dict(await request.form())
    form_data = validate_form_data(form_data)
    logger.info(f'calc form data -- {form_data}-- {len(form_data)} -- {type(form_data)}')

    form_data_ids = [x.get('nomer_punkta') for x in json.loads(form_data['source_id'].replace("\'", "\""))]
    # todo: извлечь курсы пользователя из формы - здесь
    # сверить отличия - если отличаются, расчет с введенными
    current_rates_from_form = {}
    for id in form_data_ids:
        current_rates_from_form[id] = {}
        for name in ['item__usd_kurs_prinyato', 'item__usd_kurs_vydano', 'item__eur_kurs_prinyato',
                     'item__eur_kurs_vydano', 'item__rub_kurs_prinyato', 'item__rub_kurs_vydano']:
            current_rates_from_form[id][name] = form_data[name]

    if not rates_from_form:
        request.session["rates_from_form"] = current_rates_from_form
        flag_make_proposed_kurses = True
    else:
        # saved_rates exist
        # compare old and new
        if request.session["rates_from_form"] == current_rates_from_form:
            # user did not change anything
            flag_make_proposed_kurses = False
        else:
            # user did change something
            flag_make_proposed_kurses = True
        request.session["rates_from_form"].update(current_rates_from_form)

    logger.debug(f"Session 'rates_from_form' set in calculate. Size: {sys.getsizeof(request.session['rates_from_form'])}") # ADDED LOGGING

    ''' 1. здесь
    если ввел юзер - НЕ вызывать make_proposed_kurses (flag_make_proposed_kurses=False) и присвоить proposed_kurses значение от юзера
    иначе - вызвать make_proposed_kurses (flag_make_proposed_kurses=True) '''
    # compare data_to_render [list[dict]] and form_data[dict[str, float]]

    _, data_to_render = process_before_viewing(current_rates_from_form, mode=mode, flag_make_proposed_kurses=flag_make_proposed_kurses)



    logger.info(f'rates_from_form, calc: {request.session["rates_from_form"] }')
    referer = request.headers.get("referer")
    if not referer:
        referer = ROOT_URL
    request.session['hello'] = 'hello from calculate'

    return RedirectResponse(url=referer, status_code=status.HTTP_303_SEE_OTHER)

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,)
