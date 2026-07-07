"""
Synthetic drift workload for the latency-predictor.

Drives /add_training_data_bulk on a local training_server. Emits two regimes:

  baseline  : ttft_ms = ttft_intercept + α_b·input_tokens + β_b·num_request_waiting + N(0, σ)
              tpot_ms = tpot_intercept + γ_b·num_tokens_generated + δ_b·num_request_running + N(0, σ)
  drift     : same shape, different α, β, γ, δ (workload shift / model swap / hw change).

The generator switches regimes at --drift-at seconds. While running, the
training_server's /metrics endpoint exposes ttft_coverage_percent and
tpot_coverage_percent (each a deque of 5 most-recent test-set coverage
values). A separate `scrape_metrics.py` script can poll those.

This script is intentionally dependency-light: requests + stdlib + numpy.

Usage:
  python synth_workload.py --duration 1800 --drift-at 900 --rate 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Regime:
    """Linear-noise model for synthetic ttft/tpot latency."""
    name: str
    ttft_intercept: float
    ttft_alpha_input: float      # per input token
    ttft_beta_waiting: float     # per queued request
    tpot_intercept: float
    tpot_gamma_generated: float  # per generated token
    tpot_delta_running: float    # per running request
    sigma_ttft: float
    sigma_tpot: float


BASELINE = Regime(
    name="baseline",
    ttft_intercept=20.0,
    ttft_alpha_input=0.05,
    ttft_beta_waiting=15.0,
    tpot_intercept=8.0,
    tpot_gamma_generated=0.02,
    tpot_delta_running=2.0,
    sigma_ttft=2.0,
    sigma_tpot=0.5,
)

def make_drift_regime(magnitude: float) -> Regime:
    """Scale baseline slope coefficients by `magnitude`. Intercepts and noise stay put.
    magnitude=3.0 reproduces the previous hard-coded DRIFT regime (3x input slope,
    2x queue, 2x decode-token slope, 4x running-request slope; for simplicity here
    we apply the same multiplier to every slope).

    Because the slopes multiply heavy-tailed features (lognormal input tokens,
    Pareto generated tokens), this inflates the spread/tails of a baseline-trained
    predictor's residual far more than its center: a dispersion-dominant drift."""
    return Regime(
        name=f"drift_{magnitude:g}x",
        ttft_intercept=BASELINE.ttft_intercept,
        ttft_alpha_input=BASELINE.ttft_alpha_input * magnitude,
        ttft_beta_waiting=BASELINE.ttft_beta_waiting * magnitude,
        tpot_intercept=BASELINE.tpot_intercept,
        tpot_gamma_generated=BASELINE.tpot_gamma_generated * magnitude,
        tpot_delta_running=BASELINE.tpot_delta_running * magnitude,
        sigma_ttft=BASELINE.sigma_ttft,
        sigma_tpot=BASELINE.sigma_tpot,
    )


def make_location_drift_regime(shift_sigmas: float) -> Regime:
    """Shift only the intercepts by `shift_sigmas` * baseline noise sigma; hold every
    slope and both noise sigmas fixed. This is a pure location shift of the
    predictor's residual distribution (the mean moves, the spread does not) — the
    matched failure mode for coverage and the counterpart to make_drift_regime's
    dispersion drift."""
    return Regime(
        name=f"loc_{shift_sigmas:g}sd",
        ttft_intercept=BASELINE.ttft_intercept + shift_sigmas * BASELINE.sigma_ttft,
        ttft_alpha_input=BASELINE.ttft_alpha_input,
        ttft_beta_waiting=BASELINE.ttft_beta_waiting,
        tpot_intercept=BASELINE.tpot_intercept + shift_sigmas * BASELINE.sigma_tpot,
        tpot_gamma_generated=BASELINE.tpot_gamma_generated,
        tpot_delta_running=BASELINE.tpot_delta_running,
        sigma_ttft=BASELINE.sigma_ttft,
        sigma_tpot=BASELINE.sigma_tpot,
    )


def make_noise_drift_regime(noise_magnitude: float) -> Regime:
    """Scale only the noise sigmas by `noise_magnitude`; hold every slope and both
    intercepts fixed. This is a pure dispersion shift of the predictor's residual
    (the spread grows, the mean does not) — the zero-location counterpart that
    isolates the regime where a magnitude signal (pinball) should out-detect the
    bounded coverage signal."""
    return Regime(
        name=f"disp_{noise_magnitude:g}x",
        ttft_intercept=BASELINE.ttft_intercept,
        ttft_alpha_input=BASELINE.ttft_alpha_input,
        ttft_beta_waiting=BASELINE.ttft_beta_waiting,
        tpot_intercept=BASELINE.tpot_intercept,
        tpot_gamma_generated=BASELINE.tpot_gamma_generated,
        tpot_delta_running=BASELINE.tpot_delta_running,
        sigma_ttft=BASELINE.sigma_ttft * noise_magnitude,
        sigma_tpot=BASELINE.sigma_tpot * noise_magnitude,
    )


def sample_features(rng: random.Random) -> dict:
    """Draw a single feature vector that covers the training buckets reasonably evenly."""
    # input_token_length: roughly log-normal — short prompts dominate but long tails matter
    input_tokens = int(rng.lognormvariate(mu=5.5, sigma=1.0))  # median ~245, tails to several K
    input_tokens = max(1, min(8192, input_tokens))

    # num_tokens_generated: clipped pareto — most generations short, some long
    generated = int(rng.paretovariate(alpha=1.3))
    generated = max(1, min(1024, generated))

    # num_request_waiting / running: tied to a poisson load level
    waiting = rng.choices([0, 1, 2, 3, 5, 8, 15], weights=[40, 20, 15, 10, 7, 5, 3])[0]
    running = rng.choices([1, 2, 4, 8], weights=[30, 40, 20, 10])[0]

    # kv_cache_percentage: beta-ish around 0.4
    kv_cache = max(0.0, min(1.0, rng.betavariate(2.0, 3.0)))

    # prefix_cache_score: bimodal — either small or big match
    prefix = rng.choices([0.0, 0.25, 0.5, 0.75, 0.95], weights=[40, 15, 15, 15, 15])[0]

    return {
        "kv_cache_percentage": kv_cache,
        "input_token_length": input_tokens,
        "num_request_waiting": waiting,
        "num_request_running": running,
        "num_tokens_generated": generated,
        "prefix_cache_score": prefix,
    }


def latency_from(regime: Regime, feat: dict, rng: random.Random) -> tuple[float, float]:
    ttft = (
        regime.ttft_intercept
        + regime.ttft_alpha_input * feat["input_token_length"] * (1 - feat["prefix_cache_score"])
        + regime.ttft_beta_waiting * feat["num_request_waiting"]
        + rng.gauss(0, regime.sigma_ttft)
    )
    tpot = (
        regime.tpot_intercept
        + regime.tpot_gamma_generated * feat["num_tokens_generated"]
        + regime.tpot_delta_running * feat["num_request_running"]
        + rng.gauss(0, regime.sigma_tpot)
    )
    return max(0.1, ttft), max(0.1, tpot)


def build_entry(regime: Regime, rng: random.Random) -> dict:
    feat = sample_features(rng)
    ttft_ms, tpot_ms = latency_from(regime, feat, rng)
    return {
        **feat,
        "actual_ttft_ms": ttft_ms,
        "actual_tpot_ms": tpot_ms,
    }


def post_bulk(url: str, entries: list[dict]) -> None:
    body = json.dumps({"entries": entries}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"POST failed: {resp.status}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--duration", type=float, default=1800.0, help="seconds")
    ap.add_argument("--drift-at", type=float, default=900.0, help="seconds (none = no drift)")
    ap.add_argument("--rate", type=float, default=20.0, help="samples per second")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--drift-mode",
        choices=["slope", "location"],
        default="slope",
        help="slope: multiply slope coefficients (dispersion-dominant drift); "
        "location: shift intercepts only (pure location drift)",
    )
    ap.add_argument(
        "--drift-magnitude",
        type=float,
        default=3.0,
        help="slope mode: multiplier applied to baseline slope coefficients",
    )
    ap.add_argument(
        "--intercept-shift-sigmas",
        type=float,
        default=2.0,
        help="location mode: intercept shift in units of baseline noise sigma",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    url = args.server.rstrip("/") + "/add_training_data_bulk"
    if args.drift_mode == "location":
        drift_regime = make_location_drift_regime(args.intercept_shift_sigmas)
    else:
        drift_regime = make_drift_regime(args.drift_magnitude)

    start = time.time()
    batch: list[dict] = []
    interval = args.batch_size / args.rate

    posted = 0
    current_regime = BASELINE
    drift_announced = False

    while True:
        elapsed = time.time() - start
        if elapsed >= args.duration:
            break

        if args.drift_at > 0 and elapsed >= args.drift_at and not drift_announced:
            current_regime = drift_regime
            drift_announced = True
            print(f"[{elapsed:7.1f}s] DRIFT applied (regime → {current_regime.name})", flush=True)

        for _ in range(args.batch_size):
            batch.append(build_entry(current_regime, rng))

        try:
            post_bulk(url, batch)
            posted += len(batch)
        except Exception as e:
            print(f"[{elapsed:7.1f}s] post error: {e}", file=sys.stderr, flush=True)
        batch.clear()

        if int(elapsed) % 30 == 0:
            print(f"[{elapsed:7.1f}s] regime={current_regime.name:8s} posted={posted}", flush=True)

        time.sleep(interval)

    print(f"done: posted={posted} in {time.time()-start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
