from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
import requests
import uvicorn
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select
from passlib.context import CryptContext

DATABASE_URL = "sqlite:///./elasticity_model.db"
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="token")

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
templates = Jinja2Templates(directory="templates")

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
    token: Annotated[str, Depends(OAUTH2_SCHEME)], db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # In a real application, you'd decode the JWT and extract the username
        # For this simple example, we'll just assume the token *is* the username
        username = token
    except Exception:
        raise credentials_exception
    user = db.exec(select(User).where(User.user_name == username)).first()
    if user is None:
        raise credentials_exception
    return user

# --- Dependency to get the currently active user ---
async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

# --- Authentication Endpoints ---
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)
):
    user = db.exec(select(User).where(User.user_name == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # In a real application, you'd generate a JWT here instead of just returning the username
    # For simplicity, we're using the username as a temporary "token"
    return {"access_token": user.user_name, "token_type": "bearer"}

# --- User Registration Endpoint ---
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

# --- Logout Endpoint (Client-side focused) ---
# In a stateless authentication system like JWT (which we're simulating here),
# there's no server-side logout. The client simply discards the token.
# This endpoint is more for providing a clear route in your API for the client
# to call, signaling the intent to log out.
@app.post("/logout")
async def logout():
    return {"message": "Logged out (token should be discarded by the client)"}

@app.get("/", response_class=HTMLResponse)
async def list_view(
        request: Request,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: Session = Depends(get_db)
):
    if not (current_user.user_name.lower().startswith('ca')):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this detail page.",
        )
    # placeholder_data = {"action": "list_view_accessed"}
    try:
        pass
        # requests.post(POST_ENDPOINT, json=placeholder_data)
    except requests.exceptions.ConnectionError:
        # print(f"Warning: Could not connect to POST endpoint: {POST_ENDPOINT}")
        pass
    return templates.TemplateResponse("list.html", {"request": request, "items": items})

@app.get("/{item_id:int}/", response_class=HTMLResponse)
async def detail_view(
        request: Request,
        item_id: int,
        current_user: Annotated[User, Depends(get_current_active_user)],
        db: Session = Depends(get_db)
):
    item = next((item for item in items if item["id"] == item_id), None)
    if not item:
        return HTMLResponse(content="Item not found", status_code=404)

    if (current_user.id != item_id) and not (current_user.user_name.lower().startswith('ca')):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this detail page.",
        )
    # In a real application, you would fetch the details for item_id from the database
    return {"item_id": item_id, "user_id": current_user.id, "details": f"Details for item {item_id}"}

    # placeholder_data = {"action": f"detail_view_accessed", "item_id": item_id}
    try:
        pass
        # requests.post(POST_ENDPOINT, json=placeholder_data)
    except requests.exceptions.ConnectionError:
        pass
        # print(f"Warning: Could not connect to POST endpoint: {POST_ENDPOINT}")

    placeholder_value = "This is a placeholder value."
    return templates.TemplateResponse("detail.html", {"request": request, "item": item, "placeholder_value": placeholder_value})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)