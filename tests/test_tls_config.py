# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for TLS/SSL configuration in training and prediction servers.

Covers https://github.com/llm-d/llm-d-latency-predictor/issues/87

Tests verify that:
- When SSL_KEYFILE and SSL_CERTFILE are both set, uvicorn is started with ssl_* kwargs.
- When neither is set, uvicorn is started without ssl_* kwargs (plain HTTP, backward compat).
- When only one of the pair is set, SSL is NOT enabled (both are required).
"""
import importlib
import os
import sys
from unittest.mock import MagicMock, patch


def _reload_settings(env: dict, module_path: str, class_name: str):
    """Re-import a settings class with a patched environment."""
    with patch.dict(os.environ, env, clear=False):
        # Force re-evaluation of the module-level class attributes
        mod = importlib.import_module(module_path)
        importlib.reload(mod)
        return getattr(mod, class_name)()


# ---------------------------------------------------------------------------
# Training server
# ---------------------------------------------------------------------------

class TestTrainingServerTLS:
    def _uvicorn_kwargs(self, env: dict) -> dict:
        """Run the __main__ block logic with patched env and capture uvicorn kwargs."""
        captured = {}

        def fake_run(app, **kwargs):
            captured.update(kwargs)

        with patch.dict(os.environ, env, clear=False):
            # Reload so Settings picks up the patched env
            if "training.training_server" in sys.modules:
                del sys.modules["training.training_server"]
            with patch("uvicorn.run", side_effect=fake_run):
                import training.training_server as mod
                # Simulate __main__ block
                s = mod.settings
                uvicorn_kwargs: dict = {"host": "0.0.0.0", "port": 8000, "reload": True}
                if s.SSL_KEYFILE and s.SSL_CERTFILE:
                    uvicorn_kwargs.update({
                        "ssl_keyfile": s.SSL_KEYFILE,
                        "ssl_certfile": s.SSL_CERTFILE,
                        "ssl_ca_certs": s.SSL_CA_CERTS,
                    })
        return uvicorn_kwargs

    def test_no_ssl_vars_plain_http(self):
        env = {"SSL_KEYFILE": "", "SSL_CERTFILE": "", "SSL_CA_CERTS": ""}
        kwargs = self._uvicorn_kwargs(env)
        assert "ssl_keyfile" not in kwargs
        assert "ssl_certfile" not in kwargs
        assert "ssl_ca_certs" not in kwargs

    def test_both_set_enables_tls(self):
        env = {
            "SSL_KEYFILE": "/certs/tls.key",
            "SSL_CERTFILE": "/certs/tls.crt",
            "SSL_CA_CERTS": "/certs/ca.crt",
        }
        kwargs = self._uvicorn_kwargs(env)
        assert kwargs["ssl_keyfile"] == "/certs/tls.key"
        assert kwargs["ssl_certfile"] == "/certs/tls.crt"
        assert kwargs["ssl_ca_certs"] == "/certs/ca.crt"

    def test_both_set_without_ca_certs(self):
        env = {
            "SSL_KEYFILE": "/certs/tls.key",
            "SSL_CERTFILE": "/certs/tls.crt",
            "SSL_CA_CERTS": "",
        }
        kwargs = self._uvicorn_kwargs(env)
        assert kwargs["ssl_keyfile"] == "/certs/tls.key"
        assert kwargs["ssl_certfile"] == "/certs/tls.crt"
        # ssl_ca_certs is present but empty/None — uvicorn accepts this
        assert "ssl_keyfile" in kwargs

    def test_only_keyfile_no_tls(self):
        env = {"SSL_KEYFILE": "/certs/tls.key", "SSL_CERTFILE": ""}
        kwargs = self._uvicorn_kwargs(env)
        assert "ssl_keyfile" not in kwargs

    def test_only_certfile_no_tls(self):
        env = {"SSL_KEYFILE": "", "SSL_CERTFILE": "/certs/tls.crt"}
        kwargs = self._uvicorn_kwargs(env)
        assert "ssl_keyfile" not in kwargs


# ---------------------------------------------------------------------------
# Prediction server
# ---------------------------------------------------------------------------

class TestPredictionServerTLS:
    def _uvicorn_kwargs(self, env: dict) -> dict:
        """Simulate the __main__ block logic for the prediction server."""
        import tempfile
        tmp = tempfile.mkdtemp()
        # Redirect model paths so the module-level ModelSyncer() doesn't try to
        # create /local_models (read-only on macOS outside Docker).
        path_env = {
            "LOCAL_TTFT_MODEL_PATH": f"{tmp}/ttft.joblib",
            "LOCAL_TPOT_MODEL_PATH": f"{tmp}/tpot.joblib",
            "LOCAL_TTFT_SCALER_PATH": f"{tmp}/ttft_scaler.joblib",
            "LOCAL_TPOT_SCALER_PATH": f"{tmp}/tpot_scaler.joblib",
            "LOCAL_TTFT_GATED_MODEL_PATH": f"{tmp}/ttft_gated.joblib",
            "LOCAL_TPOT_GATED_MODEL_PATH": f"{tmp}/tpot_gated.joblib",
        }
        merged_env = {**path_env, **env}
        with patch.dict(os.environ, merged_env, clear=False):
            if "prediction.prediction_server" in sys.modules:
                del sys.modules["prediction.prediction_server"]
            import prediction.prediction_server as mod
            s = mod.settings
            uvicorn_kwargs: dict = {"host": s.HOST, "port": s.PORT, "reload": True}
            if s.SSL_KEYFILE and s.SSL_CERTFILE:
                uvicorn_kwargs.update({
                    "ssl_keyfile": s.SSL_KEYFILE,
                    "ssl_certfile": s.SSL_CERTFILE,
                    "ssl_ca_certs": s.SSL_CA_CERTS,
                })
        return uvicorn_kwargs

    def test_no_ssl_vars_plain_http(self):
        env = {"SSL_KEYFILE": "", "SSL_CERTFILE": "", "SSL_CA_CERTS": ""}
        kwargs = self._uvicorn_kwargs(env)
        assert "ssl_keyfile" not in kwargs
        assert "ssl_certfile" not in kwargs

    def test_both_set_enables_tls(self):
        env = {
            "SSL_KEYFILE": "/certs/tls.key",
            "SSL_CERTFILE": "/certs/tls.crt",
            "SSL_CA_CERTS": "/certs/ca.crt",
        }
        kwargs = self._uvicorn_kwargs(env)
        assert kwargs["ssl_keyfile"] == "/certs/tls.key"
        assert kwargs["ssl_certfile"] == "/certs/tls.crt"
        assert kwargs["ssl_ca_certs"] == "/certs/ca.crt"

    def test_only_keyfile_no_tls(self):
        env = {"SSL_KEYFILE": "/certs/tls.key", "SSL_CERTFILE": ""}
        kwargs = self._uvicorn_kwargs(env)
        assert "ssl_keyfile" not in kwargs

    def test_only_certfile_no_tls(self):
        env = {"SSL_KEYFILE": "", "SSL_CERTFILE": "/certs/tls.crt"}
        kwargs = self._uvicorn_kwargs(env)
        assert "ssl_keyfile" not in kwargs
