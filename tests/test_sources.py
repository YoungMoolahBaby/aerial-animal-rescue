import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aerial_animal_rescue.sources import download_file

DATA = bytes(range(256)) * 40  # 10240 bytes


class FlakyRangeHandler(BaseHTTPRequestHandler):
    """Serves DATA with Range support; the first response is cut off halfway."""

    requests_seen: list[str | None] = []

    def do_GET(self):
        rng = self.headers.get("Range")
        self.requests_seen.append(rng)
        start = int(rng.removeprefix("bytes=").rstrip("-")) if rng else 0
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(DATA) - start))
        self.end_headers()
        body = DATA[start:]
        if len(self.requests_seen) == 1:
            body = body[: len(body) // 2]  # connection drops mid-download
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    FlakyRangeHandler.requests_seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FlakyRangeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/data.bin"
    httpd.shutdown()


def test_download_resumes_after_dropped_connection(tmp_path, server):
    dest = tmp_path / "data.bin"
    download_file(server, dest, size=len(DATA), wait_s=0)
    assert dest.read_bytes() == DATA
    assert FlakyRangeHandler.requests_seen == [None, f"bytes={len(DATA) // 2}-"]
    assert not dest.with_name("data.bin.part").exists()


def test_download_resumes_part_from_earlier_run(tmp_path, server):
    FlakyRangeHandler.requests_seen = ["earlier"]  # no drop this time
    dest = tmp_path / "data.bin"
    dest.with_name("data.bin.part").write_bytes(DATA[:1000])
    download_file(server, dest, size=len(DATA), wait_s=0)
    assert dest.read_bytes() == DATA
    assert FlakyRangeHandler.requests_seen[-1] == "bytes=1000-"


def test_download_file_rejects_short_download(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 100)
    dest = tmp_path / "out" / "src.bin"
    with pytest.raises(OSError, match="incomplete download"):
        # server promised more than it sent
        download_file(src.as_uri(), dest, size=1000, retries=2, wait_s=0)
    assert not dest.exists()
    download_file(src.as_uri(), dest, size=100)
    assert dest.read_bytes() == b"x" * 100
