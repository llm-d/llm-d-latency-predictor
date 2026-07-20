#!/usr/bin/env python3
"""Recording proxy between the EPP and the latency-predictor training server.

Sits on the wire the EPP already uses (TRAINING_SERVER_URL) and appends every
/add_training_data_bulk entry to a JSONL trace file while forwarding all
traffic unchanged. The recorded trace is the raw material for offline
feature A/B analysis (see offline_feature_ab.py).

Stdlib only -- runs in any Python 3.9+ image, no pip installs needed.

Point the EPP at this proxy:
    TRAINING_SERVER_URL=http://<recorder-host>:8002

Environment:
    FORWARD_URL  real training server base URL (default http://localhost:8000)
    TRACE_FILE   JSONL output path (default /data/trace.jsonl)
    PORT         listen port (default 8002)

Usage:
    python benchmarks/trace_recorder.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORWARD_URL = os.getenv("FORWARD_URL", "http://localhost:8000").rstrip("/")
TRACE_FILE = os.getenv("TRACE_FILE", "/data/trace.jsonl")
PORT = int(os.getenv("PORT", "8002"))

# hop-by-hop headers must not be forwarded (RFC 7230 §6.1)
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_lock = threading.Lock()
_recorded = 0


def _record_entries(payload: bytes) -> None:
    global _recorded
    try:
        entries = json.loads(payload).get("entries", [])
    except (json.JSONDecodeError, AttributeError):
        logging.warning("Unparseable bulk payload (%d bytes) — forwarded but not recorded", len(payload))
        return
    received_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        with open(TRACE_FILE, "a") as f:
            for entry in entries:
                entry["_recorded_at"] = received_at
                f.write(json.dumps(entry) + "\n")
        _recorded += len(entries)
        if _recorded % 500 < len(entries):
            logging.info("Recorded %d training entries so far", _recorded)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet per-request noise
        pass

    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        if self.path == "/recorder/healthz":
            resp = json.dumps({"status": "ok", "recorded": _recorded, "trace_file": TRACE_FILE}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if self.command == "POST" and self.path == "/add_training_data_bulk" and body:
            _record_entries(body)

        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        req = urllib.request.Request(f"{FORWARD_URL}{self.path}", data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=60) as upstream:
                data = upstream.read()
                self.send_response(upstream.status)
                for k, v in upstream.headers.items():
                    if k.lower() not in _HOP_BY_HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (urllib.error.URLError, TimeoutError) as e:
            logging.error("Upstream %s %s failed: %s", self.command, self.path, e)
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _proxy


def main() -> None:
    os.makedirs(os.path.dirname(TRACE_FILE) or ".", exist_ok=True)
    logging.info("Trace recorder on :%d → %s (trace: %s)", PORT, FORWARD_URL, TRACE_FILE)
    ThreadingHTTPServer(("0.0.0.0", PORT), ProxyHandler).serve_forever()


if __name__ == "__main__":
    main()
