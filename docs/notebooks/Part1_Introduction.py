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

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="ryqPvTKI19zH"
# # Part 1: Introduction
# The goal of this colab is to introduce the core abstractions used within this library.
# These include the `Task` and `Optimizer` objects.
#
# We will first introduce these abstractions and illustrate basic functionality. We will then show how to define a custom `Optimizer`, and how to optimize optimizers via gradient-based meta-training.
#
# This colab serves as a brief, limited introduction to the capabilities of the library. Further notebooks introduce further functionality as well as more complex learned optimizer models.

# + [markdown] id="FyfbHOeU2RL8"
# ## Prerequisites
#
# This document assumes knowledge of JAX which is covered in depth at the [JAX Docs](https://jax.readthedocs.io/en/latest/index.html).
# In particular, we would recomend making your way through [JAX tutorial 101](https://jax.readthedocs.io/en/latest/jax-101/index.html).

# + id="dLD5VE3EACSI"
# !pip install git+https://github.com/google/learned_optimization.git

# + id="HHYLexWR2BZK"
import numpy as np
import jax.numpy as jnp
import jax
from matplotlib import pylab as plt

from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule

from learned_optimization.tasks import quadratics
from learned_optimization.tasks import fixed_mlp
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.datasets import base as datasets_base

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import eval_training

import haiku as hk
import tqdm

# + [markdown] id="oX9d589m2YPv"
# ## Tasks
#
# A `Task` is an object containing a specification of a machine learning or optimization problem. The `Task` requires:
#   * Parameters: for example, these may include the decision variables of an arbitrary optimization problem, or parameters of a predictive model such as a neural network. These are initialized through the `init` method.
#   * Optionally a model state: this includes model parameters which are not to be updated via gradients. One example is the running population statistics used within batch norm.
#   * Optionally, a `.dataset` attribute with iterators of datasets.
#   * A loss function: this maps from the parameters, and possibly a batch of data to a loss.
#
# This object can be thought of as a loss function, and these are the base objects we train learned optimizers to perform well on.
#
# Tasks contain the following:
#   * A `init` function which initializes the parameters of the task.
#   * A `loss` function, which evaluates the loss given parameters and data.
#   * Optionally a `.dataset` attribute with iterators of datasets.
#
# For tasks which make use of a model state (e.g. tasks with batchnorm), a `init_with_state` and `loss_with_state` will also be provided.
#
# We'll begin by looking at some built-in tasks in the library. In future colabs, we will discuss how custom tasks can be designed, and how families of tasks can be efficiently designed for parallelization.
#
# We will look at the `FashionMnistRelu32_8` task. This task consists of a 1 hidden layer MLP trained on Fashion MNIST resized to 8x8.
#
# First, let's initialize the parameters.

# + id="xQQ-82RFIlln"
key = jax.random.PRNGKey(0)
task = fixed_mlp.FashionMnistRelu32_8()

params = task.init(key)
jax.tree_map(lambda x: x.shape, params)

# + [markdown] id="jZc2JfTxLCq8"
# We can see we initialized parameters which correspond to the weights of the MLP.
#
# Next, let's look at the data.

# + id="28fk2PfOLQWy"
batch = next(task.datasets.train)
jax.tree_map(lambda x: (x.shape, x.dtype), batch)

# + [markdown] id="nLY1PbqxLcRp"
# We get batches of 128 with images of size 8x8 and labels stored as integers.
#
# To compute losses, we can call the `loss` function. Some loss functions can be stochastic. For these, in addition to passing in params, and the batch of data, we also pass in a random number.

# + id="_w4Na8ppLP8W"
key, key1 = jax.random.split(key)

loss = task.loss(params, key1, batch)
loss

# + [markdown] id="UnSk-UqNL_PA"
# Function transformations can also be used to compute gradients.

# + id="BXKorJb4MC-u"
loss, grad = jax.value_and_grad(task.loss)(params, key1, batch)
jax.tree_map(lambda x: x.shape, grad)

# + [markdown] id="d-rzlJALMWLd"
# Now let's pull this together to train this task with SGD. Note that we will _jit_ the loss gradient computation for improved performance---if this is not familiar, we recommend reading about [Just in Time Compilation with JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html).

# + id="5ypAKqlqMf1e"
grad_fn = jax.jit(jax.value_and_grad(task.loss))
key = jax.random.PRNGKey(0)
params = task.init(key)
lr = 0.1

for i in range(1000):
  key, key1 = jax.random.split(key)
  batch = next(task.datasets.train)
  l, grads = grad_fn(params, key1, batch)
  # apply SGD to each parameter
  params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
  if i % 100 == 0:
    test_l = task.loss(params, key, next(task.datasets.test))
    print(f"train loss at {i}: {float(l)}. Test loss: {float(test_l)}")

# + [markdown] id="TytyVsF8Nol1"
# Note the evaluations in the above are quite noisy as they are only done on a single batch of data.

# + [markdown] id="jyJoJC3zOpH0"
# ## Optimizers
# We have so far implemented a rough SGD optimizer to train our model parameters. In this section, we will develop useful abstractions to create more powerful optimizers.
#
# Sadly there is no gold standard interface for optimizers in Jax: there are Flax's optimizers, optax optimizers, optimizers from jaxopt, and optix. This library uses it's own interface to expose additional types of inputs to the optimizer. These additional inputs will become more obvious when we discuss learned optimizers later in this colab, as well as in future colabs.
#
#
# In this library, optimizers are stateless classes that implement:
#   * an `init` which creates an `OptimizerState` instance which wraps parameters and optionally a model stats as well as contains any additional optimizer state needed (e.g. momentum values)
#   * a `get_params` and `get_state`  which return the parameters and state of the `OptimizerState`.
#   * an `update` function which takes in a previous optimizer state, gradients, and optionally a loss values to produce a new `OptimizerState` (with new parameters).
#
#
# Let's look at a couple examples. First, SGD:

# + id="pPzF1KD9RNLK"
fake_params = {"a": jnp.zeros((2,))}

opt = opt_base.SGD(1e-4)
opt_state = opt.init(fake_params)
opt_state

# + [markdown] id="K8ezKWoDSU_d"
# We can see the `opt_state` has parameter values, and a couple other values such as current iteration.

# + id="6AD8CTQOSaR7"
opt = opt_base.Adam(1e-4)
opt_state = opt.init(fake_params)
opt_state

# + [markdown] id="-fKs57O5SeMY"
# Adam, on the other hand, has more data inside as it contains first and second moment accumulators.
#
# Now let's take one step with an optimizer.

# + id="yOBZfwxjSjV_"
fake_grads = {"a": jnp.ones((2,))}
fake_loss = 10.

next_opt_state = opt.update(opt_state, fake_grads, fake_loss)
opt.get_params(next_opt_state)

# + [markdown] id="1j2R8-VGS0OX"
# We can see the parameters of our model have been updated slightly.
#
# Now let's pull this all together and train a Task with this optimizer API.

# + id="klycie9FS7vJ"
task = fixed_mlp.FashionMnistRelu32_8()
key = jax.random.PRNGKey(0)
params = task.init(key)

opt = opt_base.Adam(1e-2)
opt_state = opt.init(params)

for i in range(10):
  batch = next(task.datasets.train)
  key, key1 = jax.random.split(key)
  params = opt.get_params(opt_state)
  loss, grads = jax.value_and_grad(task.loss)(params, key1, batch)
  opt_state = opt.update(opt_state, grads, loss)
  print(loss)

# + [markdown] id="hJWeOzzQTs_J"
# The above doesn't make use of any sort of `jax.jit` and thus it is slow. In practice, we often like to create one update function which maps from one `opt_state` to the next and jit this entire function. For example:

# + id="A0qeh0ZWT9qD"
task = fixed_mlp.FashionMnistRelu32_8()
key = jax.random.PRNGKey(0)
params = task.init(key)

opt = opt_base.Adam(1e-2)
opt_state = opt.init(params)


@jax.jit
def update(opt_state, key, batch):
  key, key1 = jax.random.split(key)
  params, model_state = opt.get_params_state(opt_state)
  loss, grads = jax.value_and_grad(task.loss)(params, key1, batch)
  opt_state = opt.update(opt_state, grads, loss=loss)

  return opt_state, key, loss


for i in range(10):
  batch = next(task.datasets.train)
  opt_state, key, loss = update(opt_state, key, batch)
  print(loss)

# + [markdown] id="PgTqylxGRITt"
# ### Defining a custom `Optimizer`
#
# To define a custom optimizer, one simply needs to define a stateless instance of the `Optimizer` class and some pytree object with the optimizer state.
#
# As an example let's implement the momentum optimizer. As our state we will use a flax dataclass (though a simple dictionary or named tuple would also suffice).

# + id="BWGUWw1GI6-8"
import flax
from typing import Any


@flax.struct.dataclass
class MomentumOptState:
  params: Any
  model_state: Any
  iteration: jnp.ndarray
  momentums: Any


class MomentumOptimizer(opt_base.Optimizer):

  def __init__(self, lr=1e-3, momentum=0.9):
    super().__init__()
    self._lr = lr
    self._momentum = momentum

  def get_state(self, opt_state):
    return opt_state.model_state

  def get_params(self, opt_state):
    return opt_state.params

  def init(self, params, model_state=None, **kwargs):
    return MomentumOptState(
        params=params,
        model_state=model_state,
        momentums=jax.tree_map(jnp.zeros_like, params),
        iteration=jnp.asarray(0, dtype=jnp.int32))

  def update(self, opt_state, grads, loss, model_state=None, **kwargs):
    struct = jax.tree_structure(grads)
    flat_momentum = jax.tree_leaves(opt_state.momentums)
    flat_grads = jax.tree_leaves(grads)
    flat_params = jax.tree_leaves(opt_state.params)

    output_params = []
    output_momentums = []
    for m, g, p in zip(flat_momentum, flat_grads, flat_params):
      next_m = m * self._momentum + g * (1 - self._momentum)
      next_p = p - next_m * self._lr
      output_params.append(next_p)
      output_momentums.append(next_m)
    return MomentumOptState(
        params=jax.tree_unflatten(struct, output_params),
        model_state=model_state,
        iteration=opt_state.iteration + 1,
        momentums=jax.tree_unflatten(struct, output_params),
    )


opt = MomentumOptimizer(lr=1)
opt_state = opt.init({"a": 1.0, "b": 2.0})
opt.update(opt_state, {"a": -1.0, "b": 1.0}, 1.0)

# + [markdown] id="EasqQh1TYuhC"
# ## Learned Optimizers
#
# Learned optimizers are simply optimizers parameterized by some additional set of variables, often called `theta` by convention.
#
# Like before, instances of `LearnedOptimizer` should contain no immutable state.
#
# They implement 2 functions:
#   * `init` which initializes the weights of the learned optimizer (e.g. randomly as done with neural networks, or with some fixed values).
#   * `opt_fn` which takes in the parameters of the learned optimizer, and produces an `Optimizer` instance.
#
#
# One of the simplest forms of learned optimizer is a hand-designed optimizer with meta-learnable hyperparameters. Let's look at `LearnableAdam`:

# + id="UGUPaUnrZur-"
lopt = lopt_base.LearnableAdam()
theta = lopt.init(key)
theta

# + [markdown] id="nsbp3Sl7Zyct"
# We see this optimizer has 4 meta-learnable parameters corresponding to log learning rate, 2 values for beta (parameterized as the log of one minus the beta values), and log epsilon.
#
# We can access an instance of the optimizer with the opt_fn, and use that optimizer just like the ones in the previous section.

# + id="uMmrvgAyZ7_D"
opt = lopt.opt_fn(theta)
opt_state = opt.init({"p": jnp.zeros([
    2,
])})

# + [markdown] id="cMC3oBf8aLsT"
# With our optimizers split up in this way we can now write functions that are a function of the learned optimizer weights.
#
# As an example, let us define a function, `meta_loss` which is the result of applying a learned optimizer to a given problem for some number of steps.

# + id="6pm07D0haajD"
task = fixed_mlp.FashionMnistRelu32_8()
key = jax.random.PRNGKey(0)

lopt = lopt_base.LearnableAdam()


def meta_loss(theta, key, batch):
  opt = lopt.opt_fn(theta)
  key1, key = jax.random.split(key)
  param = task.init(key1)
  opt_state = opt.init(param)
  for i in range(4):
    param = opt.get_params(opt_state)
    key1, key = jax.random.split(key)
    l, grad = jax.value_and_grad(task.loss)(param, key1, batch)
    opt_state = opt.update(opt_state, grad, l)

  param, state = opt.get_params_state(opt_state)
  key1, key = jax.random.split(key)
  final_loss = task.loss(param, key1, batch)
  return final_loss


batch = next(task.datasets.train)
meta_loss(theta, key, batch)

# + [markdown] id="5B2m8ahObRik"
# But let's not stop there, we can leverage jax now to easily compute meta-gradients, or gradients with respect to the weights of the learned optimizer. This will take a bit to compile (~20 seconds on my machine) as this computation graph is a bit complex. Note: this can be greatly reduced by leveraging `jax.lax.scan`!

# + id="RzXuxJRTbWqF"
meta_value_and_grad = jax.jit(jax.value_and_grad(meta_loss))

ml, meta_grad = meta_value_and_grad(theta, key, batch)
meta_grad

# + [markdown] id="Tx35AOaYbva3"
# We can see that this meta-gradient is saying we should increase the log learning rate to improve performance.
#
# We can now meta-train by using an additional optimizer -- this time to optimize `theta`, the weights of the learned optimizer.

# + id="d0eRqFMYb8LS"
theta_opt = opt_base.Adam(1e-2)

key = jax.random.PRNGKey(0)
theta = lopt.init(key)
theta_opt_state = theta_opt.init(theta)

learning_rates = []
learnable_adam_meta_losses = []
for i in range(2000):
  batch = next(task.datasets.train)
  key, key1 = jax.random.split(key)
  theta = theta_opt.get_params(theta_opt_state)
  ml, meta_grad = meta_value_and_grad(theta, key, batch)
  theta_opt_state = theta_opt.update(theta_opt_state, meta_grad, ml)
  learning_rates.append(theta["log_lr"])
  learnable_adam_meta_losses.append(ml)
  if i % 100 == 0:
    print(ml)

# + id="sdaoU5ssc-Cu"
import numpy as np
from matplotlib import pylab as plt

plt.semilogy(np.exp(learning_rates))
plt.ylabel("learning rate")
plt.xlabel("meta-iteration")

# + [markdown] id="Xs6BY33BdSka"
# And there you have it: we have used gradient-based meta-training to train the hyperparameters of our Adam optimizer! This is the core idea in learned optimizers.
#
# Fitting a handful of scalars is a relatively simple application of the tools we have developed. In this library there are a number of more complex learned optimizers. We will explore these models, as well as more complex library functionality, in the next colab notebook.
