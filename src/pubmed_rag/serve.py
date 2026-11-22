"""Start the API in a background thread (used by the Colab notebook)."""

from __future__ import annotations

import threading
import time

import uvicorn


def start_background(app, host: str = "127.0.0.1", port: int = 8000, timeout: float = 900.0):
    """Run uvicorn in a daemon thread and block until the app (including model loading) is up.
    Returns (server, thread); set `server.should_exit = True` to stop it."""
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("server thread exited during startup; see the log above")
        if time.time() > deadline:
            raise TimeoutError(f"server did not start within {timeout:.0f}s")
        time.sleep(0.25)
    return server, thread
