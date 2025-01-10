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

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

items = [
    {"id": 1, "name": "Item A", "description": "Description for Item A"},
    {"id": 2, "name": "Item B", "description": "Description for Item B"},
    {"id": 3, "name": "Item C", "description": "Description for Item C"},
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
        return RedirectResponse(url="/login?error=Incorrect username or password", status_code=status.HTTP_303_SEE_OTHER)

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
    db_user = User(id = user.id, user_name=user.user_name, hashed_password=hashed_password)
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
    return templates.TemplateResponse("list.html", {"request": request, "items": items, "current_user": current_user})

@app.get("/{item_id}/", response_class=HTMLResponse)
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: Session = Depends(get_db),
):
    if item_id == 369:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    item = next((item for item in items if item["id"] == item_id), None)
    if not item:
        return HTMLResponse(content="Item not found", status_code=404)

    if (current_user.id != item_id) and not (current_user.user_name.lower().startswith('ca')):
        return RedirectResponse(url=f"/{current_user.id}", status_code=status.HTTP_303_SEE_OTHER)

    placeholder_value = "This is a placeholder value."
    return templates.TemplateResponse("detail.html", {"request": request, "item": item, "placeholder_value": placeholder_value, "current_user": current_user})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)