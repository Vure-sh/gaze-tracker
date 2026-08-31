"""Asynchronous threaded camera capture stream with device fallback and low-latency frame buffer."""

import time
import threading
import subprocess
from typing import Optional, Tuple, Union, List, Any
import cv2
import numpy as np


def ensure_tablet_stream(device_path: str = "/dev/video9") -> Optional[subprocess.Popen]:
    """
    Spawns scrcpy tablet camera stream in background if not already running.
    Returns subprocess.Popen handle if launched, or None.
    """
    # Check if scrcpy is already streaming to the v4l2 device
    try:
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return None
    except Exception:
        pass

    # Check if adb device is connected
    try:
        adb_check = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=2.0)
        if "device\n" in adb_check.stdout or "\tdevice" in adb_check.stdout:
            proc = subprocess.Popen([
                "scrcpy", "--video-source=camera", "--camera-id=1",
                "--camera-size=1280x720", "--camera-fps=30",
                "--video-bit-rate=12M",
                f"--v4l2-sink={device_path}", "--no-playback", "--no-audio"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
            return proc
    except Exception:
        pass

    return None


def open_camera_device(
    camera_arg: Union[str, int] = "/dev/video9",
    fallback_devices: Optional[List[Union[str, int]]] = None
) -> Tuple[Optional[cv2.VideoCapture], Union[str, int]]:
    """
    Attempts to open the specified camera index or device path with automatic fallback.
    Returns (cv2.VideoCapture, actual_device_identifier).
    """
    if fallback_devices is None:
        fallback_devices = ["/dev/video9", 0, 1, 2]

    # Priority 1: Check tablet virtual camera
    camera_str = str(camera_arg).strip()
    if camera_str in ("/dev/video9", "9", "tablet"):
        ensure_tablet_stream("/dev/video9")
        try:
            # Try integer index 9 with V4L2 first
            for dev_target in [9, "/dev/video9"]:
                cap = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)
                if cap.isOpened():
                    for _ in range(12):
                        ret, _ = cap.read()
                        if ret:
                            return cap, "/dev/video9"
                        time.sleep(0.1)
                    cap.release()
        except Exception:
            pass

    # Priority 2: Try requested argument
    try:
        idx = int(camera_arg)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, idx
            cap.release()
    except (ValueError, TypeError):
        try:
            cap = cv2.VideoCapture(str(camera_arg), cv2.CAP_V4L2)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, camera_arg
                cap.release()
        except Exception:
            pass

    # Priority 3: Auto-fallback search across candidate devices
    for test_dev in fallback_devices:
        if test_dev == "/dev/video9":
            ensure_tablet_stream("/dev/video9")
            try:
                cap = cv2.VideoCapture("/dev/video9", cv2.CAP_V4L2)
            except Exception:
                continue
        else:
            try:
                cap = cv2.VideoCapture(test_dev)
            except Exception:
                continue

        if cap.isOpened():
            for _ in range(8):
                ret, _ = cap.read()
                if ret:
                    return cap, test_dev
                time.sleep(0.1)
            cap.release()

    return None, camera_arg


class ThreadedCameraStream:
    """
    Dedicated background worker thread for non-blocking OpenCV video capture.
    Guarantees freshest frame delivery, zero queue lag, and graceful failure recovery.
    """

    def __init__(
        self,
        camera_src: Union[str, int] = "/dev/video9",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        flip_horizontal: bool = True
    ):
        self.camera_src = camera_src
        self.width = width
        self.height = height
        self.target_fps = fps
        self.flip_horizontal = flip_horizontal

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_device: Optional[Union[str, int]] = None
        self.scrcpy_process: Optional[subprocess.Popen] = None

        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._fps_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

    def start(self) -> "ThreadedCameraStream":
        """Start the background frame capture thread."""
        if self._running:
            return self

        # Open video capture device
        self.cap, self.actual_device = open_camera_device(self.camera_src)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device '{self.camera_src}' or fallback devices.")

        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraCaptureWorker")
        self._thread.start()

        # Wait up to 1.0s for the first frame
        self._new_frame_event.wait(timeout=1.0)
        return self

    def _capture_worker(self) -> None:
        """Dedicated background thread continuously polling video frames."""
        while self._running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            with self._frame_lock:
                self._frame = frame
                self._timestamp = now

            self._fps_count += 1
            elapsed = now - self._fps_start_time
            if elapsed >= 1.0:
                self._current_fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_start_time = now

            self._new_frame_event.set()

    def read(self, wait_timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest available frame from the background worker.
        Returns (has_frame, frame_copy).
        """
        if not self._running:
            return False, None

        # Wait briefly for a new frame if none has arrived yet
        if self._frame is None:
            self._new_frame_event.wait(timeout=wait_timeout)

        with self._frame_lock:
            if self._frame is None:
                return False, None
            # Return a copy or reference (return a copy to prevent race during drawing)
            frame_copy = self._frame.copy()
            self._new_frame_event.clear()
            return True, frame_copy

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read latest frame and its timestamp without blocking.
        Returns (has_frame, frame_copy, timestamp).
        """
        if not self._running:
            return False, None, 0.0

        with self._frame_lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._timestamp

    @property
    def fps(self) -> float:
        """Get rolling measured capture framerate."""
        return self._current_fps

    @property
    def is_opened(self) -> bool:
        """Check if camera capture device is actively opened."""
        return self.cap is not None and self.cap.isOpened() and self._running

    def stop(self) -> None:
        """Stop capture thread and release device resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.scrcpy_process is not None:
            try:
                self.scrcpy_process.terminate()
            except Exception:
                pass
            self.scrcpy_process = None

    def release(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "ThreadedCameraStream":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
