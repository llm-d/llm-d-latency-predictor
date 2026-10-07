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
import logging

from training.training_server import LatencyPredictor


def test_calculate_metrics_on_test_does_not_error_on_drop_timestamp(caplog):
    predictor = LatencyPredictor()
    with caplog.at_level(logging.ERROR):
        result = predictor._calculate_metrics_on_test(
            model=None,
            scaler=None,
            test_data=[{"timestamp": 1.0, "ttft": 0.5}],
            model_name="ttft",
            target_col="ttft",
        )
    assert result == (None, None, None)
    assert not any("_drop_timestamp" in record.getMessage() for record in caplog.records)
