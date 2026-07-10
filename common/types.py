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
import random
from collections import deque
from enum import StrEnum


class ModelType(StrEnum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class ObjectiveType(StrEnum):
    QUANTILE = "quantile"
    MEAN = "mean"


class QueueGatedModel:
    """Wraps noqueue + queued sub-models into one joblib-serializable object.

    At prediction time the caller checks num_request_waiting and picks the
    appropriate sub-model + scaler from inside this wrapper.
    """

    def __init__(self, noqueue_model, queued_model, noqueue_scaler=None, queued_scaler=None):
        self.noqueue_model = noqueue_model
        self.queued_model = queued_model
        self.noqueue_scaler = noqueue_scaler
        self.queued_scaler = queued_scaler


class RandomDropDeque(deque):
    def __init__(self, maxlen):
        super().__init__()
        self._maxlen = maxlen

    def append(self, item):
        if len(self) >= self._maxlen:
            idx = random.randrange(len(self))
            self.rotate(-idx)
            self.popleft()
            self.rotate(idx)
        super().append(item)

    def appendleft(self, item):
        if len(self) >= self._maxlen:
            idx = random.randrange(len(self))
            self.rotate(len(self) - idx - 1)
            self.pop()
            self.rotate(-(len(self) - idx - 1))
        super().appendleft(item)
