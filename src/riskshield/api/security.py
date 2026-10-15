"""Single-service authentication and bounded request admission."""

import hmac
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic


@dataclass(frozen=True)
class ServiceSettings:
    api_key: str = ""
    demo_mode: bool = False
    database: Path = Path("artifacts/service.sqlite3")
    requests_per_minute: int = 120
    max_body_bytes: int = 262144

    def __post_init__(self) -> None:
        if not self.demo_mode and len(self.api_key) < 24:
            raise ValueError("Set RISKSHIELD_API_KEY (24+ characters) or RISKSHIELD_DEMO_MODE=1")
        if self.requests_per_minute < 1 or self.max_body_bytes < 1:
            raise ValueError("Request limits must be positive")

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            api_key=os.getenv("RISKSHIELD_API_KEY", ""),
            demo_mode=os.getenv("RISKSHIELD_DEMO_MODE") == "1",
            database=Path(os.getenv("RISKSHIELD_DATABASE", "artifacts/service.sqlite3")),
            requests_per_minute=int(os.getenv("RISKSHIELD_REQUESTS_PER_MINUTE", "120")),
        )


class Admission:
    """One bounded global rolling window per process; no attacker-controlled key map."""

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.events: deque[float] = deque()
        self.lock = Lock()

    def authorized(self, key: str) -> bool:
        return self.settings.demo_mode or hmac.compare_digest(
            key.encode(), self.settings.api_key.encode()
        )

    def admit(self, now: float | None = None) -> bool:
        now = monotonic() if now is None else now
        with self.lock:
            while self.events and self.events[0] <= now - 60:
                self.events.popleft()
            if len(self.events) >= self.settings.requests_per_minute:
                return False
            self.events.append(now)
            return True
