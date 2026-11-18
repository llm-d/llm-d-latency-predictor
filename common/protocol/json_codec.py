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
"""JSON/HTTP compatibility for the latency protocol. Application validation is separate."""

import json
from typing import TypeVar

from google.protobuf import json_format
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message

from common.protocol import latency_pb2 as pb

_Message = TypeVar("_Message", bound=Message)
_FLAT_MESSAGES = {pb.PredictionRequest.DESCRIPTOR.full_name, pb.TrainingEntry.DESCRIPTOR.full_name}
_INT64_KINDS = {
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_SINT64,
    FieldDescriptor.TYPE_SFIXED64,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_FIXED64,
}


def _field_value(value, field: FieldDescriptor, encode: bool):
    if field.type == FieldDescriptor.TYPE_MESSAGE:
        return _transform(value, field.message_type, encode)
    if encode and field.type in _INT64_KINDS:
        # Keep the numeric JSON contract without a float conversion.
        return int(value)
    if encode and field.type in {FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT} and isinstance(value, str):
        raise ValueError(f"Non-finite number in {field.full_name}")
    return value


def _is_repeated(field: FieldDescriptor) -> bool:
    repeated = getattr(field, "is_repeated", None)
    if repeated is not None:
        return repeated
    # protobuf 5.x exposes repetition through label.
    return field.label == FieldDescriptor.LABEL_REPEATED


def _transform(value, descriptor: Descriptor, encode: bool):
    # Well-known types have their own JSON mappings, including array-valued trees.
    if descriptor.file.package == "google.protobuf" or not isinstance(value, dict):
        return value
    result = dict(value)
    if descriptor.full_name in _FLAT_MESSAGES and not encode:
        result.pop("features", None)
        features = {}
        for field in pb.Features.DESCRIPTOR.fields:
            for key in {field.name, field.json_name}:
                if key in result:
                    features[key] = result.pop(key)
        if features:
            result["features"] = features

    for field in descriptor.fields:
        for key in {field.name, field.json_name}:
            if key not in result or result[key] is None:
                continue
            item = result[key]
            if field.type == FieldDescriptor.TYPE_MESSAGE and field.message_type.GetOptions().map_entry:
                map_value = field.message_type.fields_by_name["value"]
                if isinstance(item, dict):
                    result[key] = {k: _field_value(v, map_value, encode) for k, v in item.items()}
            elif _is_repeated(field):
                if isinstance(item, list):
                    result[key] = [_field_value(v, field, encode) for v in item]
            else:
                result[key] = _field_value(item, field, encode)

    if descriptor.full_name in _FLAT_MESSAGES and encode:
        result.update(result.pop("features", {}))
    return result


def to_dict(message: Message) -> dict:
    """Return snake_case, flat HTTP fields with integer-valued JSON numbers."""
    data = json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return _transform(data, message.DESCRIPTOR, encode=True)


def from_dict(data: dict, message: _Message) -> _Message:
    """Parse HTTP fields, ignoring unknown fields during mixed-version rollout."""
    if not isinstance(data, dict):
        raise json_format.ParseError("Expected a JSON object")
    parsed = type(message)()
    json_format.ParseDict(_transform(data, message.DESCRIPTOR, encode=False), parsed, ignore_unknown_fields=True)
    message.CopyFrom(parsed)
    return message


def marshal(message: Message) -> bytes:
    """Encode an HTTP request or model-artifact object."""
    return json.dumps(to_dict(message), separators=(",", ":"), allow_nan=False).encode()


def unmarshal(data: str | bytes, message: _Message) -> _Message:
    """Decode an HTTP request or model-artifact object."""
    return from_dict(json.loads(data), message)
