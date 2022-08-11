---
jupytext:
  formats: ipynb,md:myst,py
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "20970d65"}

# Part 2: Custom Tasks, Task Families, and Performance Improvements

In this part, we will look at how to define custom tasks and datasets. We will also consider _families_ of tasks, which are common specifications of meta-learning problems. Finally, we will look at how to efficiently parallelize over tasks during training.

+++ {"id": "ef075664"}

## Prerequisites

This document assumes knowledge of JAX which is covered in depth at the [JAX Docs](https://jax.readthedocs.io/en/latest/index.html).
In particular, we would recomend making your way through [JAX tutorial 101](https://jax.readthedocs.io/en/latest/jax-101/index.html). We also recommend that you have worked your way through Part 1.

```{code-cell}
:id: f560fa24

!pip install git+https://github.com/google/learned_optimization.git
```

```{code-cell}
---
executionInfo:
  elapsed: 24640
  status: ok
  timestamp: 1643173374165
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 04db154b
---
import numpy as np
import jax.numpy as jnp
import jax
from matplotlib import pylab as plt

from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule

from learned_optimization.tasks import quadratics
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.datasets import base as datasets_base

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import eval_training

import haiku as hk
import tqdm
```

+++ {"id": "707298d0"}

## Defining a custom Dataset

The dataset's in this library consists of iterators which yield batches of the corresponding data. For the provided tasks, these dataset have 4 splits of data rather than the traditional 3. We have "train" which is data used by the task to train a model, "inner_valid" which contains validation data for use when inner training (training an instance of a task). This could be use for, say, picking hparams. "outer_valid" which is used to meta-train with -- this is unseen in inner training and thus serves as a basis to train learned optimizers against. "test" which can be used to test the learned optimizer with.

To make a dataset, simply write 4 iterators with these splits.

For performance reasons, creating these iterators cannot be slow.
The existing dataset's make extensive use of caching to share iterators across tasks which use the same data iterators.
To account for this reuse, it is expected that these iterators are always randomly sampling data and have a large shuffle buffer so as to not run into any sampling issues.

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1643173374354
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: df73c83b
outputId: 435d2986-d008-412e-bd71-bb7d9c404f3d
---
import numpy as np


def data_iterator():
  bs = 3
  while True:
    batch = {"data": np.zeros([bs, 5])}
    yield batch


@datasets_base.dataset_lru_cache
def get_datasets():
  return datasets_base.Datasets(
      train=data_iterator(),
      inner_valid=data_iterator(),
      outer_valid=data_iterator(),
      test=data_iterator())


ds = get_datasets()
next(ds.train)
```

+++ {"id": "410f2024"}

## Defining a custom `Task`

To define a custom class, one simply needs to write a base class of `Task`. Let's look at a simple task consisting of a quadratic task with noisy targets.

```{code-cell}
---
executionInfo:
  elapsed: 799
  status: ok
  timestamp: 1643173375359
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 27dbabeb
outputId: 394bb22e-4481-4ee6-8d3b-490d1b77f35c
---
# First we construct data iterators.
def noise_datasets():

  def _fn():
    while True:
      yield np.random.normal(size=[4, 2]).astype(dtype=np.float32)

  return datasets_base.Datasets(
      train=_fn(), inner_valid=_fn(), outer_valid=_fn(), test=_fn())


class MyTask(tasks_base.Task):
  datasets = noise_datasets()

  def loss(self, params, rng, data):
    return jnp.sum(jnp.square(params - data))

  def init(self, key):
    return jax.random.normal(key, shape=(4, 2))


task = MyTask()
key = jax.random.PRNGKey(0)
key1, key = jax.random.split(key)
params = task.init(key)

task.loss(params, key1, next(task.datasets.train))
```

+++ {"id": "a16e5e3b"}

## Meta-training on multiple tasks: `TaskFamily`

What we have shown previously was meta-training on a single task instance.
While sometimes this is sufficient for a given situation, in many situations we seek to meta-train a meta-learning algorithm such as a learned optimizer on a mixture of different tasks.

One path to do this is to simply run more than one meta-gradient computation, each with different tasks, average the gradients, and perform one meta-update.
This works great when the tasks are quite different -- e.g. meta-gradients when training a convnet vs a MLP.
A big negative to this is that these meta-gradient calculations are happening sequentially, and thus making poor use of hardware accelerators like GPU or TPU.

As a solution to this problem, we have an abstraction of a `TaskFamily` to enable better use of hardware. A `TaskFamily` represents a distribution over a set of tasks and specifies particular samples from this distribution as a pytree of jax types.

The function to sample these configurations is called `sample`, and the function to get a task from the sampled config is `task_fn`. `TaskFamily` also optionally contain datasets which are shared for all the `Task` it creates.

As a simple example, let's consider a family of quadratics parameterized by meansquared error to some point which itself is sampled.

```{code-cell}
---
executionInfo:
  elapsed: 64
  status: ok
  timestamp: 1643173375565
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: f1c7d7f8
---
PRNGKey = jnp.ndarray
TaskParams = jnp.ndarray


class FixedDimQuadraticFamily(tasks_base.TaskFamily):
  """A simple TaskFamily with a fixed dimensionality but sampled target."""

  def __init__(self, dim: int):
    super().__init__()
    self._dim = dim
    self.datasets = None

  def sample(self, key: PRNGKey) -> TaskParams:
    # Sample the target for the quadratic task.
    return jax.random.normal(key, shape=(self._dim,))

  def task_fn(self, task_params: TaskParams) -> tasks_base.Task:
    dim = self._dim

    class _Task(tasks_base.Task):

      def loss(self, params, rng, _):
        # Compute MSE to the target task.
        return jnp.sum(jnp.square(task_params - params))

      def init(self, key):
        return jax.random.normal(key, shape=(dim,))

    return _Task()
```

+++ {"id": "37652293"}

*With* this task family defined, we can create instances by sampling a configuration and creating a task. This task acts like any other task in that it has an `init` and a `loss` function.

```{code-cell}
---
executionInfo:
  elapsed: 334
  status: ok
  timestamp: 1643173376069
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: fba3b113
outputId: 1f62a1e6-8c99-4991-b2d7-380c5adee83a
---
task_family = FixedDimQuadraticFamily(10)
key = jax.random.PRNGKey(0)
task_cfg = task_family.sample(key)
task = task_family.task_fn(task_cfg)

key1, key = jax.random.split(key)
params = task.init(key)
batch = None
task.loss(params, key, batch)
```

+++ {"id": "8b25914f"}

To achive speedups, we can now leverage `jax.vmap` to train *multiple* task instances in parallel! Depending on the task, this can be considerably faster than serially executing them.

```{code-cell}
---
executionInfo:
  elapsed: 1508
  status: ok
  timestamp: 1643173377718
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 7dded1ea
outputId: d75a21c5-0210-4482-f088-1b5a0ce92c17
---
def train_task(cfg, key):
  task = task_family.task_fn(cfg)
  key1, key = jax.random.split(key)
  params = task.init(key1)
  opt = opt_base.Adam()

  opt_state = opt.init(params)

  for i in range(4):
    params = opt.get_params(opt_state)
    loss, grad = jax.value_and_grad(task.loss)(params, key, None)
    opt_state = opt.update(opt_state, grad, loss=loss)
  loss = task.loss(params, key, None)
  return loss


task_cfg = task_family.sample(key)
print("single loss", train_task(task_cfg, key))

keys = jax.random.split(key, 32)
task_cfgs = jax.vmap(task_family.sample)(keys)
losses = jax.vmap(train_task)(task_cfgs, keys)
print("multiple losses", losses)
```

+++ {"id": "79f74adc"}

Because of this ability to apply vmap over task families, this is the main building block for a number of the high level libraries in this package. Single tasks can always be converted to a task family with:

```{code-cell}
---
executionInfo:
  elapsed: 3041
  status: ok
  timestamp: 1643173380925
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 6cd2f682
---
single_task = image_mlp.ImageMLP_FashionMnist8_Relu32()
task_family = tasks_base.single_task_to_family(single_task)
```

+++ {"id": "905293c1"}

This wrapper task family has no configuable value and always returns the base task.

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1643173381121
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: cb049afb
outputId: 47250a96-577d-4d74-de88-b21d17f27fa3
---
cfg = task_family.sample(key)
print("config only contains a dummy value:", cfg)
task = task_family.task_fn(cfg)
# Tasks are the same
assert task == single_task
```

+++ {"id": "760f8e76"}

## Limitations of `TaskFamily`
Task families are designed for, and only work for variation that results in a static computation graph. This is required for `jax.vmap` to work.

This means things like naively changing hidden sizes, or number of layers, activation functions is off the table.

In some cases, one can leverage other jax control flow such as `jax.lax.cond` to select between implementations. For example, one could make a `TaskFamily` that used one of 2 activation functions. While this works, the resulting vectorized computation could be slow and thus profiling is required to determine if this is a good idea or not.

In this code base, we use `TaskFamily` to mainly parameterize over different kinds of initializations.
