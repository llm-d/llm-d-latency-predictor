# Copyright 2026 The llm-d Authors.
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
import copy
import json
import math
from pathlib import Path

import pytest
from google.protobuf import json_format

from common.protocol import latency_pb2 as pb
from common.protocol.json_codec import from_dict, marshal, to_dict, unmarshal


@pytest.fixture(scope="module")
def wire():
    return json.loads((Path(__file__).parent / "testdata/latency-wire.json").read_text())


@pytest.mark.parametrize(
    ("case", "message_type"),
    [
        ("prediction_zero", pb.PredictionRequest),
        ("prediction_extended", pb.PredictionRequest),
        ("prediction_legacy", pb.PredictionRequest),
        ("training_ttft", pb.TrainingEntry),
        ("training_tpot", pb.TrainingEntry),
        ("training_legacy", pb.TrainingEntry),
        ("coefficients", pb.ModelCoefficients),
        ("trees", pb.XGBoostTrees),
        ("buckets", pb.BucketCounts),
        ("model_info", pb.ModelInfo),
        ("export_metadata", pb.ModelExportMetadata),
    ],
)
def test_wire_round_trip(wire, case, message_type):
    original = copy.deepcopy(wire[case])
    message = from_dict(wire[case], message_type())
    assert to_dict(message) == original
    assert json.loads(marshal(message)) == original
    assert unmarshal(json.dumps(original), message_type()) == message
    assert wire[case] == original


def test_optional_zero_is_distinct_from_missing(wire):
    legacy = from_dict(wire["prediction_legacy"], pb.PredictionRequest())
    zero = from_dict(wire["prediction_zero"], pb.PredictionRequest())
    assert not legacy.features.HasField("encoder_input_size")
    assert zero.features.HasField("encoder_input_size")
    assert zero.features.encoder_input_size == 0
    assert "encoder_input_size" not in to_dict(legacy)
    assert to_dict(zero)["encoder_input_size"] == 0


def test_unknown_fields_are_ignored_without_changing_known_values(wire):
    data = {**wire["training_legacy"], "future_feature": 123, "future_metadata": "value"}
    message = from_dict(data, pb.TrainingEntry())
    assert to_dict(message) == wire["training_legacy"]
    assert data["future_feature"] == 123


@pytest.mark.parametrize(
    ("container_type", "field", "case"),
    [
        (pb.BulkPredictionRequest, "requests", "prediction_extended"),
        (pb.BulkTrainingRequest, "entries", "training_ttft"),
    ],
)
def test_bulk_requests_keep_flat_rows(wire, container_type, field, case):
    data = {field: [wire[case], wire[case]]}
    message = from_dict(data, container_type())
    assert to_dict(message) == data
    assert len(getattr(message, field)) == 2
    assert "features" not in to_dict(message)[field][0]


def test_metadata_does_not_become_a_feature(wire):
    entry = from_dict(wire["training_ttft"], pb.TrainingEntry())
    assert entry.HasField("predicted_ttft_ms")
    assert not entry.HasField("predicted_tpot_ms")
    metadata_names = {key for field in pb.TrainingEntry.DESCRIPTOR.fields for key in (field.name, field.json_name)}
    feature_names = {key for field in pb.Features.DESCRIPTOR.fields for key in (field.name, field.json_name)}
    assert metadata_names.isdisjoint(feature_names)


def test_integer_values_are_not_rounded_or_quoted(wire):
    message = from_dict(wire["prediction_extended"], pb.PredictionRequest())
    assert message.features.prefill_tokens_in_flight == 9007199254740993
    assert message.features.decode_tokens_in_flight == 9223372036854775807
    encoded = json.loads(marshal(message))
    assert type(encoded["prefill_tokens_in_flight"]) is int
    assert encoded["prefill_tokens_in_flight"] == 9007199254740993


def test_invalid_integer_is_rejected(wire):
    data = {**wire["prediction_legacy"], "input_token_length": 1.5}
    with pytest.raises(json_format.ParseError):
        from_dict(data, pb.PredictionRequest())


def test_nested_features_are_not_an_alternative_http_shape(wire):
    data = {**wire["prediction_legacy"], "features": {"input_token_length": 999}}
    assert to_dict(from_dict(data, pb.PredictionRequest())) == wire["prediction_legacy"]


def test_reusing_a_message_clears_previous_values(wire):
    message = from_dict(wire["prediction_extended"], pb.PredictionRequest())
    from_dict(wire["prediction_legacy"], message)
    assert not message.features.HasField("encoder_input_size")
    assert to_dict(message) == wire["prediction_legacy"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_numbers_are_not_sent_as_json_strings(value):
    message = pb.PredictionRequest(features=pb.Features(kv_cache_percentage=value))
    with pytest.raises(ValueError):
        marshal(message)


def test_invalid_input_does_not_replace_message(wire):
    message = from_dict(wire["prediction_legacy"], pb.PredictionRequest())
    with pytest.raises(json_format.ParseError):
        from_dict({"input_token_length": 1.5}, message)
    assert to_dict(message) == wire["prediction_legacy"]
