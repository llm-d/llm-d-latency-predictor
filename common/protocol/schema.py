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
"""Application rules attached to the shared latency protocol."""

import json
from functools import lru_cache

from google.protobuf.descriptor import Descriptor, FieldDescriptor

from common.protocol import latency_pb2 as pb


@lru_cache(maxsize=256)
def rules_for(field: FieldDescriptor) -> pb.FieldRules:
    return field.GetOptions().Extensions[pb.field_rules]


@lru_cache(maxsize=256)
def default_for(field: FieldDescriptor):
    rules = rules_for(field)
    return json.loads(rules.default_json) if rules.HasField("default_json") else field.default_value


def http_fields(descriptor: Descriptor):
    for field in descriptor.fields:
        if field.message_type == pb.Features.DESCRIPTOR:
            yield from field.message_type.fields
        else:
            yield field


@lru_cache(maxsize=32)
def required_numeric_fields(descriptor: Descriptor) -> tuple[str, ...]:
    return tuple(field.name for field in http_fields(descriptor) if rules_for(field).required)
