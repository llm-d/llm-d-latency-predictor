def test_training_replaces_cold_start_models(tmp_path, monkeypatch):
    """A retrain must replace the one-point cold-start models."""
    monkeypatch.setenv("LATENCY_MODEL_TYPE", "xgboost")
    monkeypatch.setenv("LATENCY_ENSEMBLE_MODE", "false")

    from training import training_server as server

    monkeypatch.setattr(server.settings, "TTFT_MODEL_PATH", str(tmp_path / "ttft.joblib"))
    monkeypatch.setattr(server.settings, "TPOT_MODEL_PATH", str(tmp_path / "tpot.joblib"))
    monkeypatch.setattr(server.settings, "TTFT_SCALER_PATH", str(tmp_path / "ttft_scaler.joblib"))
    monkeypatch.setattr(server.settings, "TPOT_SCALER_PATH", str(tmp_path / "tpot_scaler.joblib"))
    monkeypatch.setattr(server.settings, "MIN_SAMPLES_FOR_RETRAIN", 10)
    monkeypatch.setattr(server.settings, "MIN_SAMPLES_FOR_RETRAIN_FRESH", 10)
    monkeypatch.setattr(server.settings, "ENSEMBLE_MODE", False)

    predictor = server.LatencyPredictor()
    predictor.load_models()

    features = {
        "kv_cache_percentage": 0.5,
        "input_token_length": 400,
        "num_request_waiting": 3,
        "num_request_running": 2,
        "num_tokens_generated": 10,
        "prefix_cache_score": 0.5,
    }
    initial = predictor.predict(features)[:2]

    for i in range(100):
        input_tokens = 50 + i
        predictor.add_training_sample(
            {
                "kv_cache_percentage": 0.2 + (i % 5) * 0.1,
                "input_token_length": input_tokens,
                "num_request_waiting": i % 4,
                "num_request_running": 1 + i % 3,
                "actual_ttft_ms": 100.0 + input_tokens * 2.0,
                "actual_tpot_ms": 20.0 + input_tokens * 0.5,
                "num_tokens_generated": 1 + i % 10,
                "prefix_cache_score": (i % 10) / 10.0,
            }
        )

    predictor.train()
    updated = predictor.predict(features)[:2]

    assert predictor.last_retrain_time is not None
    assert updated != initial
    assert updated != (10.0, 10.0)
    assert predictor.is_ready
