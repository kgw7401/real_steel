"""Camera input module. Captures frames from webcam via OpenCV."""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, shape (H, W, 3)
    timestamp: float  # time.time()
    frame_number: int


class Camera:
    def __init__(
        self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None
        self.frame_count = 0

        # Threading state
        self._lock = threading.Lock()
        self._latest_frame: Frame | None = None
        self._frame_changed = False
        self._running = False
        self._thread: threading.Thread | None = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Start background capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

        return True

    def _capture_thread(self) -> None:
        while self._running:
            if self.cap is None:
                break
            ret, image = self.cap.read()
            if not ret:
                continue
            self.frame_count += 1
            frame = Frame(
                image=image,
                timestamp=time.time(),
                frame_number=self.frame_count,
            )
            with self._lock:
                self._latest_frame = frame
                self._frame_changed = True

    def read(self) -> tuple[Frame | None, bool]:
        """Return (frame, is_new) where is_new indicates the frame hasn't been read before."""
        with self._lock:
            frame = self._latest_frame
            is_new = self._frame_changed
            self._frame_changed = False
            return frame, is_new

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
