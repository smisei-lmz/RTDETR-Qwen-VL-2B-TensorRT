"""Protocol and scheduling rules shared by ROS nodes and dependency-free tests."""
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Callable, Optional


class Status(str, Enum):
    OK = "OK"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    STALE = "STALE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class FrameKey:
    session_id: str
    frame_id: int


@dataclass(frozen=True)
class Snapshot:
    key: FrameKey
    captured_ns: int
    image: bytes
    mime: str = "image/jpeg"
    clock_domain: str = "ros"
    timestamp_source: str = "host_acquire"


@dataclass(frozen=True)
class Request:
    request_id: str
    question: str
    snapshot: Snapshot
    max_new_tokens: int = 64
    timeout_s: float = 15.0
    max_input_age_s: float = 3.0
    max_answer_age_s: float = 20.0


@dataclass(frozen=True)
class Answer:
    request_id: str
    key: FrameKey
    captured_ns: int
    status: Status
    text: str
    model: str
    elapsed_s: float
    reason: str = ""


def source_age_s(snapshot: Snapshot, now_source_ns: int) -> float:
    return (now_source_ns - snapshot.captured_ns) / 1e9


def matches_request(answer: Answer, request: Request) -> bool:
    return (answer.request_id == request.request_id and answer.key == request.snapshot.key
            and answer.captured_ns == request.snapshot.captured_ns)


def validate_request(req: Request, now_source_ns: int) -> None:
    if not req.request_id or len(req.request_id) > 128:
        raise ValueError("invalid request_id")
    if not req.question.strip() or len(req.question) > 2000:
        raise ValueError("question must contain 1..2000 characters")
    if not req.snapshot.key.session_id or req.snapshot.key.frame_id < 0:
        raise ValueError("missing or invalid frame identity")
    if req.snapshot.clock_domain != "ros":
        raise ValueError("this demo requires a shared ROS clock domain")
    if req.snapshot.mime not in ("image/jpeg", "image/png"):
        raise ValueError("unsupported image MIME type")
    if not 0 < len(req.snapshot.image) <= 1024 * 1024:
        raise ValueError("image must contain 1 byte..1 MiB")
    if not 1 <= req.max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be 1..256")
    for name, value, cap in (("timeout", req.timeout_s, 120),
                             ("max_input_age", req.max_input_age_s, 10),
                             ("max_answer_age", req.max_answer_age_s, 180)):
        if not math.isfinite(value) or not 0 < value <= cap:
            raise ValueError(f"invalid {name}")
    age = source_age_s(req.snapshot, now_source_ns)
    if age < -0.05 or age > req.max_input_age_s:
        raise ValueError("input timestamp is stale or in the future; check clocks")


class LatestSlot:
    """One owned value; overwrite only the pending value, never an active task."""
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self.replaced = 0

    def put(self, value):
        with self._lock:
            old = self._value
            self._value = value
            if old is not None:
                self.replaced += 1
            return old

    def take(self):
        with self._lock:
            value, self._value = self._value, None
            return value

    def peek(self):
        with self._lock:
            return self._value


class SingleFlight:
    """Reserve before accepting a ROS action, release only after backend returns."""
    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False

    def reserve(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def release(self):
        with self._lock:
            if not self._busy:
                raise RuntimeError("double release")
            self._busy = False


class StopControl:
    """Cooperative cancellation. Does not promise to preempt an executing kernel."""
    def __init__(self, timeout_s: float, cancelled: Callable[[], bool],
                 monotonic: Callable[[], float] = time.monotonic):
        self._clock = monotonic
        self.started = monotonic()
        self.deadline = self.started + timeout_s
        self._cancelled = cancelled

    def reason(self) -> Optional[Status]:
        if self._cancelled():
            return Status.CANCELLED
        if self._clock() >= self.deadline:
            return Status.TIMEOUT
        return None

    def elapsed(self):
        return self._clock() - self.started


def run_request(req: Request, backend, cancelled: Callable[[], bool],
                now_source_ns: Callable[[], int],
                feedback: Callable[[str, float], None] = lambda phase, elapsed: None) -> Answer:
    stop = StopControl(req.timeout_s, cancelled)
    text, reason = "", ""
    try:
        validate_request(req, now_source_ns())
        if stop.reason() is None:
            feedback("generating", stop.elapsed())
            text = backend.generate(req, stop, feedback)
        status = stop.reason() or Status.OK
        age = source_age_s(req.snapshot, now_source_ns())
        if status == Status.OK and (age < -0.05 or age > req.max_answer_age_s):
            status, reason = Status.STALE, "image aged out or ROS clock jumped"
        if status != Status.OK:
            text = ""  # Partial/cancelled output is not a successful answer.
    except Exception as exc:
        status = stop.reason() or Status.ERROR
        reason = f"{type(exc).__name__}: {exc}"[:500]
        text = ""
    return Answer(req.request_id, req.snapshot.key, req.snapshot.captured_ns,
                  status, text, backend.name, stop.elapsed(), reason)
