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

"""Tests for serialization_utils."""

import os
import pickle

import numpy as np
import pytest

from learned_optimization import serialization_utils


def test_msgpack_serialization_integrity():
    """Verify that msgpack preserves data integrity for common types."""
    original_state = {
        "params": np.array([1.0, 2.0], dtype=np.float32),
        "step": 42,
        "metadata": {"task": "mnist", "active": True},
    }
    packed = serialization_utils.safe_pack(original_state)
    unpacked = serialization_utils.safe_unpack(packed)

    assert unpacked["step"] == original_state["step"]
    assert np.array_equal(unpacked["params"], original_state["params"])
    assert unpacked["metadata"]["task"] == "mnist"


def test_msgpack_rejects_pickle_payload():
    """Security test: Ensure msgpack doesn't accidentally execute pickle data."""

    class Malicious:

        def __reduce__(self):
            # If executed, this would create a file.
            return (open, ("rce_detected.txt", "w"))

    pickle_payload = pickle.dumps(Malicious())

    with pytest.raises(Exception):
        # Msgpack should fail to decode a pickle stream
        serialization_utils.safe_unpack(pickle_payload)

    assert not os.path.exists("rce_detected.txt")


def test_msgpack_complex_objects_serialization():
    """Verify that objects with __dict__ are converted to dicts (safe mode)."""

    class Dummy:

        def __init__(self, x):
            self.x = x

    obj = Dummy(10)
    packed = serialization_utils.safe_pack(obj)
    unpacked = serialization_utils.safe_unpack(packed)

    # In our safe implementation, complex classes are converted to dicts
    # to avoid arbitrary class instantiation.
    assert isinstance(unpacked, dict)
    assert unpacked["__type__"] == "Dummy"
    assert unpacked["data"]["x"] == 10
