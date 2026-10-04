"""Session repository interface and in-memory implementation."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from neural_forge.core.exceptions import SessionNotFoundError, SessionStateError
from neural_forge.schemas.sessions import SessionState


class SessionRepository(ABC):
    """Abstract storage for examination sessions, allowing a Redis implementation later."""

    @abstractmethod
    async def create(self, state: SessionState) -> None:
        """Store a new session."""

    @abstractmethod
    async def get(self, session_id: str) -> SessionState:
        """Return a copy of a stored session or raise SessionNotFoundError."""

    @abstractmethod
    async def save(self, state: SessionState) -> None:
        """Replace the stored state of an existing session."""

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session and all of its data; return whether it existed."""

    @abstractmethod
    async def count(self) -> int:
        """Return the number of stored sessions."""

    @abstractmethod
    async def purge_expired(self) -> int:
        """Delete sessions that have not been updated within the time-to-live."""


class InMemorySessionRepository(SessionRepository):
    """Process-local session storage with time-to-live expiry and per-session isolation."""

    def __init__(self, ttl_seconds: int, max_sessions: int) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create(self, state: SessionState) -> None:
        """Store a new session, enforcing the capacity limit."""
        await self.purge_expired()
        async with self._lock:
            if state.session_id in self._sessions:
                raise SessionStateError(f"Session {state.session_id} already exists")
            if len(self._sessions) >= self._max_sessions:
                raise SessionStateError("The server is at session capacity; please try again later.")
            self._sessions[state.session_id] = state.model_copy(deep=True)

    async def get(self, session_id: str) -> SessionState:
        """Return a deep copy so callers cannot mutate stored state accidentally."""
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or self._is_expired(state):
                raise SessionNotFoundError(f"Session {session_id} was not found")
            return state.model_copy(deep=True)

    async def save(self, state: SessionState) -> None:
        """Replace stored state and refresh the update time."""
        async with self._lock:
            if state.session_id not in self._sessions:
                raise SessionNotFoundError(f"Session {state.session_id} was not found")
            updated = state.model_copy(deep=True)
            updated.updated_at = datetime.now(timezone.utc)
            self._sessions[state.session_id] = updated

    async def delete(self, session_id: str) -> bool:
        """Remove a session and all its stored findings and report."""
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def count(self) -> int:
        """Return the number of stored sessions."""
        async with self._lock:
            return len(self._sessions)

    async def purge_expired(self) -> int:
        """Delete expired sessions and return how many were removed."""
        async with self._lock:
            expired_ids = [session_id for session_id, state in self._sessions.items() if self._is_expired(state)]
            for session_id in expired_ids:
                del self._sessions[session_id]
            return len(expired_ids)

    def _is_expired(self, state: SessionState) -> bool:
        return datetime.now(timezone.utc) - state.updated_at > self._ttl
