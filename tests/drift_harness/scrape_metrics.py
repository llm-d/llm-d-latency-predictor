"""
Polls the training_server's /metrics endpoint, extracts the
ttft_coverage_percent and tpot_coverage_percent values (each a deque of up
to 5 most-recent test-set coverage measurements), and appends one row per
poll to a CSV.

Output columns:
  elapsed_sec, ttft_cov_latest, ttft_cov_mean5, tpot_cov_latest, tpot_cov_mean5

Each coverage value is a fraction in [0, 1] (the actual fraction of test
samples for which y_true <= y_pred). For a p90 model the well-calibrated
value is 0.9.

Usage:
  python scrape_metrics.py --duration 1800 --output coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.request

COV_RE = re.compile(
    r'(?P<name>(?:ttft|tpot)_coverage_percent)\{idx="(?P<idx>\d+)"\}\s+(?P<value>[\d.eE+-]+)'
)
QL_RE = re.compile(
    r'(?P<name>(?:ttft|tpot)_quantile_loss)\{idx="(?P<idx>\d+)"\}\s+(?P<value>[\d.eE+-]+)'
)


def fetch_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def _parse_series(body: str, pattern: re.Pattern, ttft_name: str) -> tuple[list[float], list[float]]:
    """Return (ttft_values, tpot_values) for a metric family, ordered by idx (oldest→newest)."""
    ttft: dict[int, float] = {}
    tpot: dict[int, float] = {}
    for m in pattern.finditer(body):
        idx = int(m["idx"])
        value = float(m["value"])
        if m["name"] == ttft_name:
            ttft[idx] = value
        else:
            tpot[idx] = value
    return (
        [ttft[i] for i in sorted(ttft)],
        [tpot[i] for i in sorted(tpot)],
    )


def parse_coverage(body: str) -> tuple[list[float], list[float]]:
    return _parse_series(body, COV_RE, "ttft_coverage_percent")


def parse_quantile_loss(body: str) -> tuple[list[float], list[float]]:
    return _parse_series(body, QL_RE, "ttft_quantile_loss")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--duration", type=float, default=1800.0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--output", default="coverage.csv")
    args = ap.parse_args()

    url = args.server.rstrip("/") + "/metrics"
    start = time.time()

    def latest(xs):
        return xs[-1] if xs else ""

    def mean5(xs):
        return sum(xs) / len(xs) if xs else ""

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "elapsed_sec",
            "ttft_cov_latest", "ttft_cov_mean5", "tpot_cov_latest", "tpot_cov_mean5",
            "ttft_ql_latest", "ttft_ql_mean5", "tpot_ql_latest", "tpot_ql_mean5",
        ])
        while True:
            elapsed = time.time() - start
            if elapsed > args.duration:
                break
            try:
                body = fetch_metrics(url)
                ttft_c, tpot_c = parse_coverage(body)
                ttft_q, tpot_q = parse_quantile_loss(body)
            except Exception as e:
                print(f"[{elapsed:7.1f}s] scrape error: {e}", file=sys.stderr, flush=True)
                ttft_c, tpot_c, ttft_q, tpot_q = [], [], [], []
            w.writerow([
                f"{elapsed:.1f}",
                latest(ttft_c), mean5(ttft_c), latest(tpot_c), mean5(tpot_c),
                latest(ttft_q), mean5(ttft_q), latest(tpot_q), mean5(tpot_q),
            ])
            f.flush()
            time.sleep(args.interval)

    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
