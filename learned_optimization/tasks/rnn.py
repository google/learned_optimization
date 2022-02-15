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

"""Recurrent RNN haiku modules."""
from typing import Optional

import haiku as hk
import jax


class IRNN(hk.VanillaRNN):
  """Identity initialized RNN.

  This was introduced in https://arxiv.org/abs/1504.00941.
  """

  def __init__(
      self,
      hidden_size: int,
      double_bias: bool = True,
      name: Optional[str] = None,
      gain: float = 1.0,
  ):
    """Constructs a Identity RNN core.

    Args:
      hidden_size: Hidden layer size.
      double_bias: Whether to use a bias in the two linear layers. This changes
        nothing to the learning performance of the cell. However, doubling will
        create two sets of bias parameters rather than one.
      name: Name of the module.
      gain: multiplier on recurrent weight identity initialization.
    """
    super().__init__(
        hidden_size=hidden_size, double_bias=double_bias, name=name)
    self.gain = gain

  def __call__(self, inputs, prev_state):
    input_to_hidden = hk.Linear(self.hidden_size)
    hidden_to_hidden = hk.Linear(
        self.hidden_size,
        with_bias=self.double_bias,
        w_init=hk.initializers.Identity(self.gain))
    out = jax.nn.relu(input_to_hidden(inputs) + hidden_to_hidden(prev_state))
    return out, out
