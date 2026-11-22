import socket
import urllib.request

from pubmed_rag.api import create_app
from pubmed_rag.serve import start_background


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_background_server_starts_serves_and_stops(assistant, settings):
    port = _free_port()
    server, thread = start_background(create_app(settings, assistant), port=port, timeout=30)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            assert r.status == 200 and b'"status":"ok"' in r.read()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert not thread.is_alive()
