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
"""FastAPI request models derived once from protobuf descriptors."""

from datetime import UTC, datetime
from typing import ClassVar

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from pydantic import BaseModel, Field, create_model

from common.protocol.json_codec import from_dict
from common.protocol.schema import default_for, http_fields, rules_for


class ProtocolModel(BaseModel):
    _message_type: ClassVar[type[Message]]

    def to_proto(self) -> Message:
        return from_dict(self.model_dump(mode="json"), self._message_type())


def request_model(
    message_type: type[Message], name: str, module: str, *, exclude: tuple[str, ...] = ()
) -> type[ProtocolModel]:
    fields = {}
    for field in http_fields(message_type.DESCRIPTOR):
        rules = rules_for(field)
        if rules.api_hidden or field.name in exclude:
            continue
        if field.message_type is not None:
            if field.message_type.full_name != "google.protobuf.Timestamp":
                raise TypeError(f"Unsupported HTTP field {field.full_name}")
            annotation = datetime
        elif field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
            annotation = float
        elif field.type == FieldDescriptor.TYPE_STRING:
            annotation = str
        else:
            annotation = int
        if rules.nullable:
            annotation = annotation | None
        constraints = {}
        if rules.HasField("minimum"):
            constraints["ge"] = rules.minimum
        if rules.HasField("maximum"):
            constraints["le"] = rules.maximum
        if rules.default_now:
            definition = Field(default_factory=lambda: datetime.now(UTC), **constraints)
        else:
            definition = Field(... if rules.required else default_for(field), **constraints)
        fields[field.name] = (annotation, definition)
    model = create_model(name, __base__=ProtocolModel, __module__=module, **fields)
    model._message_type = message_type
    return model
