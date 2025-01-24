from typing import Optional, Dict
import uuid
from starlette.middleware.sessions import SessionMiddleware, SessionStore
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

# --- 1. Define a Server-Side Session Storage ---

class InMemorySessionStorage(SessionStore):
    """
    In-memory session storage. **Not suitable for production** as data is lost on server restart
    and not shared between processes/instances. Good for development and demonstration.

    For production, consider using a persistent storage like a database or file system.
    """
    def __init__(self):
        self._sessions: Dict[str, dict] = {}

    async def load_session(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)

    async def save_session(self, session_id: str, data: dict) -> bool:
        if not data:  # Delete session if data is empty
            return await self.delete_session(session_id)
        self._sessions[session_id] = data
        return True

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def clear(self): # Optional: Method to clear all sessions (for testing/admin)
        self._sessions.clear()


# --- 2. Create a Custom Session Middleware ---

class ServerSideSessionMiddleware(SessionMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        secret_key: str,
        session_cookie: str = "server_session",  # Custom cookie name
        session_path: str = "/",
        max_age: Optional[int] = None,
        same_site: str = "lax",
        https_only: bool = True,
        domain: Optional[str] = None,
        store: Optional[SessionStore] = None, # Accept custom store
    ) -> None:
        if store is None:
            store = InMemorySessionStorage() # Default to in-memory if not provided
        super().__init__(
            app,
            secret_key=secret_key,
            session_cookie=session_cookie,
            session_path=session_path,
            max_age=max_age,
            same_site=same_site,
            https_only=https_only,
            domain=domain,
            store=store, # Pass the custom store to the base class
        )


    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        session_id: Optional[str] = request.cookies.get(self.session_cookie)
        if not session_id:
            session_id = str(uuid.uuid4()) # Generate new session ID if not found in cookie

        store = self.store # Access the store instance from the middleware

        session_data = await store.load_session(session_id) or {} # Load from server-side store
        request.state.session = session_data

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                response_session = request.state.session
                await store.save_session(session_id, response_session) # Save session data to server-side store

                # Set the session cookie in the response
                headers = message.get("headers", [])
                header_set_cookie = (
                    b"set-cookie",
                    self.session_cookie.encode()
                    + b"="
                    + session_id.encode()
                    + self.cookie_header(request),
                )
                headers.append(header_set_cookie)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)
