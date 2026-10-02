from training.training_server import LatencyPredictor, TrainingEntry, settings


def test_timestamped_samples_train_and_persist_models(monkeypatch, tmp_path):
    for name in (
        "TTFT_MODEL_PATH",
        "TPOT_MODEL_PATH",
        "TTFT_SCALER_PATH",
        "TPOT_SCALER_PATH",
        "TTFT_GATED_MODEL_PATH",
        "TPOT_GATED_MODEL_PATH",
    ):
        monkeypatch.setattr(settings, name, str(tmp_path / f"{name}.joblib"))
    monkeypatch.setattr(settings, "MIN_SAMPLES_FOR_RETRAIN", 10)
    monkeypatch.setattr(settings, "OBJECTIVE_TYPE", "mean")
    monkeypatch.setattr(settings, "TEST_TRAIN_RATIO", 0)

    predictor = LatencyPredictor(model_type="xgboost")
    predictor.load_models()
    rows = [
        TrainingEntry(
            kv_cache_percentage=0.2,
            input_token_length=tokens,
            num_request_waiting=0,
            num_request_running=1,
            actual_ttft_ms=tokens * 2,
            actual_tpot_ms=20 if tokens == 100 else 80,
            num_tokens_generated=50,
            prefix_cache_score=0,
        ).model_dump()
        for tokens in [100] * 80 + [10000] * 80
    ]
    predictor.add_training_samples(rows)
    predictor.ttft_test_data.extend([rows[0], rows[-1]])
    predictor.tpot_test_data.extend([rows[0], rows[-1]])
    predictor.train()

    assert predictor.last_retrain_time is not None
    restored = LatencyPredictor(model_type="xgboost")
    restored.load_models()
    short_ttft, short_tpot, *_ = restored.predict(rows[0])
    long_ttft, long_tpot, *_ = restored.predict(rows[-1])
    assert short_ttft < long_ttft
    assert short_tpot < long_tpot
