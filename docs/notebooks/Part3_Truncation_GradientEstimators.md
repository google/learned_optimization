---
jupytext:
  formats: ipynb,md:myst,py
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "Tfc1UZvtOpLJ"}

# Part 3: Truncation and GradientEstimator

```{code-cell}
:id: c-TW0zBs3ggj

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

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import training
from learned_optimization import eval_training

import haiku as hk
import tqdm
```

+++ {"id": "2--dyixmsdZg"}

## Training learned optimizers: Truncated training, and different kinds of gradient estimators.

In the the previous colabs we showed some of the core abstractions this library is based on: `Task`, `Optimizer`, `LearnedOptimizer`, and `TaskFamily`. We also showed a rather minimal meta-training procedure
which trains learned optimizers to perform well with a handful of inner-training steps.

In this colab, we will discuss meta-training more in detail and discuss some of the more heavy weight abstractions designed to facilitate meta-training.
This includes methods built on truncated backprop through time, evolutionary strategies, and persistent evolutionary strategies.

This will be divided into three main components: `TruncationSchedule` which define how truncated training works, `GradientEstimator` which often leverage truncations and estimate meta-gradients -- or gradients of the meta-loss with respect to the learned optimizer weights, and `GradientLearner` which take these gradients and manage updating the learned optimizer weights.

+++ {"id": "teJagHCT3pfE"}

## Truncated training and TruncationSchedules

When applying a learned optimizer to train some target task, one usually wants the optimizer to be performant for a very large number of steps as training a model can take hundreds to hundreds of thousands of iterations.
Ideally we would like our meta-training procedure to mirror the testing setup but given how long these unrolls (iterative application of the learned optimizer) can be this can become challenging. Truncated training is one solution to this. The core idea is to never run an entire inner-problem to completion, but instead unroll a shorter segment, and leverage information from that shorter segment to update the weights of the learned optimizer.

This is most commonly seen in the form of truncated backpropogation through time and is used to train training recurrent neural networks. More recently, truncated training has been used to train RL algorithms (e.g. A3C).

Truncated training has a number of benifits. First, it greatly reduces the amount of computation needed before updating the learned optimizer. If one has length 100 truncations for a total length of 10k iterations, one 100x more updates to the weights of the learned optimizer. For some methods, like PES, we can even do these gradient estimates in an unbiased way (technically less biased, see PES paper for a discussion on hysteresis). For others, such as gradient based meta-training, and other ES variants, this comes at the cost of bias.

In code, truncations are handed by a TruncationSchedule subclass which is a small, stateless classes which manage how long we should be computing training for. For example, here we see a constant length truncation which runs for 10 steps then reports done.

```{code-cell}
---
executionInfo:
  elapsed: 57
  status: ok
  timestamp: 1643170875395
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 5QM56Si43pS7
outputId: 0ea08631-fafb-4478-91bd-739d1e47ecdd
---
trunc_sched = truncation_schedule.ConstantTruncationSchedule(10)
outer_state = None
key = jax.random.PRNGKey(0)
trunc_state = trunc_sched.init(key, outer_state)
for i in range(12):
  trunc_state, is_done = trunc_sched.next_state(trunc_state, i, key,
                                                outer_state)
  print(i, is_done)
```

+++ {"id": "TgAjP4qL9koL"}

In practice, we often run these sequentially.

For example, here is a loop which let's us sequentially train a model over and over again.
To do this, we must keep track of some state which progresses from inner-iteration to
inner-iteration. In this case, this is the problem we are training's opt_state, the
state of the truncation, and a rng key.

```{code-cell}
---
executionInfo:
  elapsed: 9594
  status: ok
  timestamp: 1643170885144
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: PQFGy7Ib9q7f
---
key = jax.random.PRNGKey(0)
task = image_mlp.ImageMLP_FashionMnist8_Relu32()
opt = opt_base.Adam(3e-3)
outer_state = None


def init_state(key):
  key, key1, key2 = jax.random.split(key, 3)
  p = task.init(key1)
  opt_state = opt.init(p)

  trunc_sched = truncation_schedule.ConstantTruncationSchedule(50)
  trunc_state = trunc_sched.init(key2, outer_state)

  return opt_state, trunc_state, key


def next_trunc_and_train(train_state_and_batch):
  (opt_state, trunc_state, key), batch = train_state_and_batch
  # progress one step on the truncation state
  trunc_state, is_done = trunc_sched.next_state(trunc_state,
                                                opt_state.iteration, key,
                                                outer_state)

  # progress one step on the
  p = opt.get_params(opt_state)
  key, key1 = jax.random.split(key)
  l, g = jax.value_and_grad(task.loss)(p, key1, batch)
  opt_state = opt.update(opt_state, g, loss=l)

  return (opt_state, trunc_state, key), is_done, l


def reset_trunc_and_init(train_state_and_batch):
  (opt_state, trunc_state, key), batch = train_state_and_batch

  # new inner problem
  p = task.init(key)
  opt_state = opt.init(p)

  # new truncation
  key, key1 = jax.random.split(key)
  trunc_state = trunc_sched.init(key1, outer_state)

  return (opt_state, trunc_state, key), False, jnp.nan


is_done = False
state = init_state(key)

losses = []
for i in range(200):
  batch = next(task.datasets.train)
  state, is_done, loss = jax.lax.cond(is_done, reset_trunc_and_init,
                                      next_trunc_and_train, (state, batch))
  losses.append(loss)
```

+++ {"id": "GcolDnWEAkOV"}

Now we can plot the losses. We see a sequence of decreasing losses, which get reset back to initialization every 25 steps.

```{code-cell}
---
colab:
  height: 296
executionInfo:
  elapsed: 265
  status: ok
  timestamp: 1643170885557
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: eGD7UTfMASBl
outputId: 9f278e50-e84e-4911-9ecc-7b63959f251f
---
plt.plot(losses)
plt.xlabel("all inner-iteration")
plt.ylabel("loss")
```

+++ {"id": "uLXf29yTA3XE"}

One key reason why this abstraction is important is when thinking about training multiple models in parallel. Naively, we could train all models starting from initialization with the exact same iteration into training.
This is sometimes refered to as "lock step" training.

One alternative is to break this lock-steping, and let our models train different parts of the inner-problem at different times.
With this TruncationState abstraction we can do this by either training with variable length unrolls, or faking the initial state so models reset early.

For starters, we can see training multiple models with lock step unrolls:

```{code-cell}
---
executionInfo:
  elapsed: 4518
  status: ok
  timestamp: 1643170890300
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: nEbvhc1-BnRH
---
n_tasks = 8


@jax.jit
@jax.vmap
def update(state, is_done, batch):
  state, is_done, loss = jax.lax.cond(is_done, reset_trunc_and_init,
                                      next_trunc_and_train, (state, batch))
  return state, is_done, loss


keys = jax.random.split(key, n_tasks)
states = jax.vmap(init_state)(keys)

losses = []
is_done = jnp.zeros([n_tasks])
for i in range(200):
  vec_batch = training.vec_get_batch(task, n_tasks)
  states, is_done, l = update(states, is_done, vec_batch)
  losses.append(l)
```

```{code-cell}
---
colab:
  height: 296
executionInfo:
  elapsed: 220
  status: ok
  timestamp: 1643170890664
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: Sm7i1JkPC0T5
outputId: 0dc0d7e0-6957-40b2-8c33-67c13d757fb0
---
losses = np.asarray(losses)
for i in range(n_tasks):
  plt.plot(losses[:, i])

plt.xlabel("all inner-iteration")
plt.ylabel("loss")
```

+++ {"id": "n5Ai-p5SDafX"}

And then, we can compare this to what the curves look like if we break the lockstep by simply faking the initial iteration.

```{code-cell}
---
executionInfo:
  elapsed: 1752
  status: ok
  timestamp: 1643170892634
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: TK_7qVf0DaDT
---
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, n_tasks)
states = jax.vmap(init_state)(keys)
opt_state, trunc_state, keys = states
import dataclasses

opt_state = opt_state._replace(
    iteration=jax.random.randint(key, [n_tasks], 0, 50))
states = (opt_state, trunc_state, keys)

losses = []
is_done = jnp.zeros([n_tasks])
for i in range(200):
  vec_batch = training.vec_get_batch(task, n_tasks)
  states, is_done, l = update(states, is_done, vec_batch)
  losses.append(l)
```

```{code-cell}
---
colab:
  height: 265
executionInfo:
  elapsed: 218
  status: ok
  timestamp: 1643170893008
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 8avrxbDDEQs_
outputId: c69a107f-c5ea-4f0b-8fde-99643b6dfcb2
---
losses = np.asarray(losses)
for i in range(n_tasks):
  plt.plot(losses[:, i])
```

+++ {"id": "FWjUQFwjEXgY"}

With this version, all the tasks start lockstep-ed initially, but then reset early so as to ensure the training trajectories are not running in lock step.

+++ {"id": "ZifT1JoXeWY5"}

## GradientEstimators

+++ {"id": "0iM51dVseYZk"}

Gradient estimators provide an interface to estimate gradients of some loss with respect to the parameters of the learned optimizer.

`learned_optimization` supports a handfull of estimators each with different strengths and weaknesses. Understanding which estimators are right for which situations is an open research question. After providing some introductions to the GradientEstimator class, we provide a quick tour of the different estimators implemented here.


All optimizers discussed here operate on a `TaskFamily` and meta-train with multiple task samples in parallel.


The `GradientEstimator` base class signature is below.

```{code-cell}
---
executionInfo:
  elapsed: 81
  status: ok
  timestamp: 1643170893389
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 9UI2k2uAhVUP
---
from typing import Optional, Tuple, Mapping
from learned_optimization.tasks import base as tasks_base
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.outer_trainers.gradient_learner import WorkerWeights, GradientEstimatorState, GradientEstimatorOut

PRNGKey = jnp.ndarray


class GradientEstimator:
  task_family: tasks_base.TaskFamily
  learned_opt: lopt_base.LearnedOptimizer

  def init_worker_state(self, worker_weights: WorkerWeights,
                        key: PRNGKey) -> GradientEstimatorState:
    raise NotImplementedError()

  def compute_gradient_estimate(
      self, worker_weights: WorkerWeights, key: PRNGKey,
      state: GradientEstimatorState, with_summary: Optional[bool]
  ) -> Tuple[GradientEstimatorOut, Mapping[str, jnp.ndarray]]:
    raise NotImplementedError()
```

+++ {"id": "e3LP8MWZhqO6"}

A gradient estimator must have an instance of a TaskFamily -- or the task that is being used to estimate gradients with, an `init_worker_state` function -- which initializes the current state of the gradient estimator, and a `compute_gradient_estimate` function which takes state and computes a bunch of outputs (`GradientEstimatorOut`) which contain the computed gradients with respect to the learned optimizer, meta-loss values, and various other information about the unroll. Additionally a mapping which contains various metrics is returned.

Both of these methods take in a `WorkerWeights` instance. This particular piece of data represents the learnable weights needed to compute a gradients. In every case this contains the weights of the learned optimizer (often called theta), but it can also contain some other state. For example if the learned optimizer has batchnorm it could also contain running averages.

+++ {"id": "IsfRHPaK-80z"}

### FullES

The FullES estimator is one of the simplest, and most reliable estimator but can be slow in practice as it does not make use of truncations. Instead, it uses antithetic sampling to estimate a gradient via ES of an entire optimization (hense the full in the name).

First we define a meta-objective, $f(\theta)$, which could be the loss at the end of training, or average loss. Next, we compute a gradient estimate via ES gradient estimation:

$\nabla_\theta f \approx \dfrac{\epsilon}{2\sigma^2} ((\theta + \epsilon) - f(\theta - \epsilon))$

We can instantiate one of these as follows:

```{code-cell}
---
executionInfo:
  elapsed: 5
  status: ok
  timestamp: 1643170893745
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: W5vQVk7o_VDq
---
task_family = quadratics.FixedDimQuadraticFamily(10)
lopt = lopt_base.LearnableAdam()
max_length = 1000

gradient_estimator = full_es.FullES(
    task_family=task_family, learned_opt=lopt, num_tasks=4, unroll_length=100)
```

```{code-cell}
---
executionInfo:
  elapsed: 58
  status: ok
  timestamp: 1643170894042
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: nLOAKLXX_nX4
---
key = jax.random.PRNGKey(0)
theta = lopt.init(key)
worker_weights = gradient_learner.WorkerWeights(
    theta=theta,
    theta_model_state=None,
    outer_state=gradient_learner.OuterState(0))
```

+++ {"id": "6Mmm0894_poZ"}

Because we are working with full lenght unrolls, this gradient estimator has no state -- there is nothing to keep track of truncation to truncation.

```{code-cell}
---
executionInfo:
  elapsed: 5
  status: ok
  timestamp: 1643170894200
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: zyaPyPLY_nX5
outputId: 7c3a6473-2c04-48ce-a0b2-4da5f2c3c58d
---
gradient_estimator_state = gradient_estimator.init_worker_state(
    worker_weights, key=key)
gradient_estimator_state
```

+++ {"id": "VwBwRmmw_zin"}

Gradients can be computed with the `compute_gradient_estimate` method.

```{code-cell}
---
executionInfo:
  elapsed: 3449
  status: ok
  timestamp: 1643170897825
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: rbSr9tFc_vth
---
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=gradient_estimator_state, with_summary=False)
```

```{code-cell}
---
executionInfo:
  elapsed: 4
  status: ok
  timestamp: 1643170898016
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: XuoeYAt9_1hL
outputId: 7a9df483-f98d-4af7-b8b1-68b5a2ec34ed
---
out.grad
```

+++ {"id": "tjiUWowcwJ1f"}

### TruncatedPES

Truncated Persistent Evolutionary Strategies (PES) is a unbiased truncation method based on ES. It was proposed in [Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies](https://arxiv.org/abs/2112.13835) and has been a promising tool for training learned optimizers.

```{code-cell}
---
executionInfo:
  elapsed: 58
  status: ok
  timestamp: 1643170898252
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: ailS8_Jbr8CT
---
task_family = quadratics.FixedDimQuadraticFamily(10)
lopt = lopt_base.LearnableAdam()
max_length = 1000
trunc_sched = truncation_schedule.LogUniformLengthSchedule(
    min_length=100, max_length=max_length)

gradient_estimator = truncated_pes.TruncatedPES(
    task_family=task_family,
    learned_opt=lopt,
    trunc_sched=trunc_sched,
    num_tasks=4,
    trunc_length=50,
    random_initial_iteration_offset=max_length)
```

```{code-cell}
---
executionInfo:
  elapsed: 58
  status: ok
  timestamp: 1643170898454
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: Nx1BTPIG4gEJ
---
key = jax.random.PRNGKey(0)
theta = lopt.init(key)
worker_weights = gradient_learner.WorkerWeights(
    theta=theta,
    theta_model_state=None,
    outer_state=gradient_learner.OuterState(0))
```

```{code-cell}
---
executionInfo:
  elapsed: 1140
  status: ok
  timestamp: 1643170899742
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: NlCnF8LT4HBx
outputId: 4dfc512b-cce4-463f-f6b4-b9939f5a0b26
---
gradient_estimator_state = gradient_estimator.init_worker_state(
    worker_weights, key=key)
```

+++ {"id": "EvCBA9Z541sn"}

Now let's look at what this state contains.

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1643170899948
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: u1QQxUYf31fy
outputId: 549c6209-6c2e-4219-d6ff-9d333c4ec313
---
jax.tree_map(lambda x: x.shape, gradient_estimator_state)
```

+++ {"id": "6meGBWzt45KV"}

First, this contains 2 instances of SingleState -- one for the positive perturbation, and one for the negative perturbation. Each one of these contains all the necessary state required to keep track of the training run. This means the opt_state, details from the truncation, the task parameters (sample from the task family), the inner_step, and a bool to determine if done or not.

We can compute one gradient estimate as follows.

```{code-cell}
---
executionInfo:
  elapsed: 4715
  status: ok
  timestamp: 1643170904965
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: MSpQTFc45lz2
outputId: 57caf27d-8317-4c8b-e93a-47b5320ac3cc
---
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=gradient_estimator_state, with_summary=False)
```

+++ {"id": "vFDZSW5h6Iri"}

This `out` object contains various outputs from the gradient estimator including gradients with respect to the learned optimizer, as well as the next state of the training models.

```{code-cell}
---
executionInfo:
  elapsed: 5
  status: ok
  timestamp: 1643170905123
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 74AnlkqB4xCV
outputId: 433091f9-6842-436f-9ac6-eb6a40438584
---
out.grad
```

```{code-cell}
---
executionInfo:
  elapsed: 63
  status: ok
  timestamp: 1643170905365
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: 82oSxk2i5-3L
outputId: 95ba9386-fac5-4295-972c-cc514da1572d
---
jax.tree_map(lambda x: x.shape, out.unroll_state)
```

+++ {"id": "MLqCPmkx6cja"}

One could simply use these gradients to meta-train, and then use the unroll_states as the next state passed into the compute gradient estimate. For example:

```{code-cell}
---
executionInfo:
  elapsed: 63
  status: ok
  timestamp: 1643170905582
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 480
id: VTPgbwtj6X0I
outputId: df510426-8627-4462-dbd8-ba7a27d3882f
---
print("Progress on inner problem before", out.unroll_state.pos_state.inner_step)
out, metrics = gradient_estimator.compute_gradient_estimate(
    worker_weights, key=key, state=out.unroll_state, with_summary=False)
print("Progress on inner problem after", out.unroll_state.pos_state.inner_step)
```

+++ {"id": "-cvH5MBs8kXb"}

We can see that for each problem we progress 50 steps (or reset and start over from scratch).
