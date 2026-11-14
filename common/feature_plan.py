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
"""Feature selection and preparation shared by training and prediction."""

from functools import lru_cache

import numpy as np
import pandas as pd
from google.protobuf.descriptor import FieldDescriptor

from common.protocol import latency_pb2 as pb
from common.protocol.schema import default_for, rules_for


@lru_cache(maxsize=16)
def _fields(target: str, encoder: bool, tokens_in_flight: bool):
    if target not in ("ttft", "tpot"):
        raise ValueError(f"Unknown prediction target {target}")
    enabled = {"encoder": encoder, "tokens_in_flight": tokens_in_flight}
    fields = []
    for field in pb.Features.DESCRIPTOR.fields:
        rules = rules_for(field)
        order = getattr(rules, f"{target}_order")
        if not order or (rules.feature_group and not enabled[rules.feature_group]):
            continue
        fields.append((order, field))
    fields.sort(key=lambda item: item[0])
    if len({order for order, _ in fields}) != len(fields):
        raise ValueError(f"Duplicate raw feature order for {target}")
    return tuple(field for _, field in fields)


def raw_fields(target: str, settings):
    return _fields(target, settings.ENABLE_ENCODER_FEATURES, settings.ENABLE_TOKEN_IN_FLIGHT_FEATURES)


def feature_columns(target: str, settings, noqueue: bool = False) -> list[str]:
    columns = [(0, "is_queued")]
    for field in raw_fields(target, settings):
        rules = rules_for(field)
        columns.append((getattr(rules, f"{target}_order"), rules.model_column or field.name))
    if target == "ttft":
        columns.extend([(100, "effective_input_tokens"), (110, "prefill_score_bucket")])
    columns.sort()
    if len({order for order, _ in columns}) != len(columns):
        raise ValueError(f"Duplicate feature order for {target}")
    return [name for _, name in columns if not noqueue or name not in ("is_queued", "num_request_waiting")]


def prepare_features(df: pd.DataFrame, target: str, settings, prefix_buckets: int = 4) -> pd.DataFrame:
    for field in raw_fields(target, settings):
        if field.name not in df:
            if rules_for(field).required:
                raise ValueError(f"Missing required feature: {field.name}")
            df[field.name] = default_for(field)
    df["pod_type_cat"] = pd.Categorical(df["pod_type"].fillna(""), categories=["", "prefill", "decode"], ordered=False)
    df["is_queued"] = (df["num_request_waiting"] > 0).astype(int)
    if target == "ttft":
        df["effective_input_tokens"] = (1 - df["prefix_cache_score"]) * df["input_token_length"]
        bucket = (df["prefix_cache_score"].clip(0, 1) * prefix_buckets).astype(int).clip(upper=prefix_buckets - 1)
        df["prefill_score_bucket"] = pd.Categorical(bucket, categories=list(range(prefix_buckets)), ordered=True)
    return df[feature_columns(target, settings)]


def raw_frames(rows: list[dict], settings, *, plain_prediction: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for target in ("ttft", "tpot"):
        frames.append(
            pd.DataFrame(
                [
                    {
                        field.name: (
                            default_for(field)
                            if plain_prediction and rules_for(field).feature_group == "tokens_in_flight"
                            else row.get(field.name, default_for(field))
                        )
                        for field in raw_fields(target, settings)
                    }
                    for row in rows
                ]
            )
        )
    return tuple(frames)


def raw_frames_fast(requests: list, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = {}
    selected = {field.name: field for target in ("ttft", "tpot") for field in raw_fields(target, settings)}
    count = len(requests)
    for name, field in selected.items():
        if field.type == FieldDescriptor.TYPE_STRING:
            columns[name] = [getattr(request, name) for request in requests]
        else:
            columns[name] = np.fromiter((getattr(request, name) for request in requests), dtype=np.float64, count=count)
    return tuple(
        pd.DataFrame({field.name: columns[field.name] for field in raw_fields(target, settings)})
        for target in ("ttft", "tpot")
    )


def training_frame(rows: list[dict], target: str, settings) -> pd.DataFrame:
    fields = raw_fields(target, settings)
    label = f"actual_{target}_ms"
    clean = [
        {
            **{
                field.name: row.get(field.name, None if rules_for(field).required else default_for(field))
                for field in fields
            },
            label: row.get(label),
        }
        for row in rows
    ]
    frame = pd.DataFrame(clean, columns=[field.name for field in fields] + [label])
    for field in fields:
        if not rules_for(field).required:
            frame[field.name] = frame[field.name].fillna(default_for(field))
    return frame.dropna()


def align_features(frame: pd.DataFrame, model, scaler=None) -> pd.DataFrame:
    owner = scaler if scaler is not None else model
    expected = getattr(owner, "feature_names_in_", None)
    if expected is None:
        expected = getattr(owner, "feature_name_", None)
    if expected is None and hasattr(owner, "get_booster"):
        expected = owner.get_booster().feature_names
    if expected is None:
        count = getattr(owner, "n_features_in_", None)
        if count is not None and count != len(frame.columns):
            raise ValueError("Model has no feature names and its feature count does not match the active feature plan")
        return frame
    expected = list(expected)
    if list(frame.columns) == expected:
        return frame
    missing = set(expected) - set(frame.columns)
    optional = {
        rules_for(field).model_column or field.name: default_for(field)
        for field in pb.Features.DESCRIPTOR.fields
        if not rules_for(field).required
    }
    if missing - optional.keys():
        raise ValueError(f"Model requires unknown feature columns: {sorted(missing - optional.keys())}")
    for name in missing:
        frame[name] = optional[name]
    return frame[expected]


def validate_model_features(model, target: str, settings, scaler=None, noqueue: bool = False):
    """Check an artifact before publishing it to prediction workers."""
    row = {field.name: default_for(field) for field in raw_fields(target, settings)}
    frame = prepare_features(pd.DataFrame([row]), target, settings)
    frame = frame[feature_columns(target, settings, noqueue=noqueue)]
    if scaler is not None:
        frame = frame.drop(columns=["prefill_score_bucket"], errors="ignore")
        frame = pd.get_dummies(frame, columns=["pod_type_cat"], prefix="pod_type", drop_first=False)
    aligned = align_features(frame, model, scaler)
    if scaler is not None:
        columns = len(aligned.columns)
        for owner in (scaler, model):
            if getattr(owner, "n_features_in_", columns) != columns:
                raise ValueError("Ridge model/scaler feature dimension mismatch")
        if np.shape(model.coef_) != (columns,):
            raise ValueError("Ridge coefficient dimension does not match its scaler")
