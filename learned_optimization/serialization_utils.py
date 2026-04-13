# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialization utilities for safe data persistence."""

import io
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np

# Patch msgpack to handle numpy types globally
m.patch()


def _encode_ext(obj):
    """Custom encoder for types not supported by msgpack."""
    if hasattr(obj, "__dict__"):
        # Handle generic objects/dataclasses by converting to dict
        return {"__type__": obj.__class__.__name__, "data": obj.__dict__}
    return obj


def safe_pack(obj: Any) -> bytes:
    """Pack an object into a secure msgpack binary format.

    Args:
      obj: The object to serialize.

    Returns:
      A bytes object representing the serialized data.
    """
    return msgpack.packb(obj, default=_encode_ext, use_bin_type=True)


def safe_unpack(data: bytes) -> Any:
    """Unpack an object from a secure msgpack binary format.

    Args:
      data: The bytes object to deserialize.

    Returns:
      The reconstructed Python object.
    """
    # For now returns raw dicts for custom types to avoid arbitrary class instantiation.
    # This is the "Security by Design" part: we don't auto-instantiate classes.
    return msgpack.unpackb(data, raw=False)
