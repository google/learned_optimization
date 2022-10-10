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

+++ {"id": "a7QBk71UMifh"}

# Part 5: Meta-training with GradientLearner

```{code-cell}
:id: MimfK6lp0vq9

import numpy as np
import jax.numpy as jnp
import jax
from matplotlib import pylab as plt

from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import lopt_truncated_step

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

+++ {"id": "vlBAN6agtoXy"}

In this notebook we build upon the previous notebook which discussed `TruncationSchedule` and `GradientEstimator`. Here, we leverage these components to provide more complete training functionality.
We do not provide a full fledged training program -- only the computations that would need to be performed each iteration of meta-training.

This code is **not** the only way to use the previous abstractions but can be convenient to avoid writing the same code over and over again. We encourage readers to write their own meta-training loops before diving into these abstractions.

+++ {"id": "a5iG2qpC8ttu"}

## Single machine meta-training: `SingleMachineGradientLearner`

The `SingleMachineGradientLearner` provides the functionality to meta-train a learned optimizer with one or more gradient estimator. As an example, let's train a learned optimizer leveraging gradients from 2 different tasks: a quadratic task, and a fashion mnist mlp.

```{code-cell}
:id: txx2_SmH8NhL

theta_opt = opt_base.Adam(1e-3)

lopt = mlp_lopt.MLPLOpt()
max_length = 300
trunc_sched = truncation_schedule.LogUniformLengthSchedule(
    min_length=100, max_length=max_length)


def grad_est_fn(task_family):
  truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family,
      lopt,
      trunc_sched,
      num_tasks=4,
      random_initial_iteration_offset=max_length)
  return truncated_pes.TruncatedPES(
      truncated_step=truncated_step, trunc_length=50)


mlp_task_family = tasks_base.single_task_to_family(
    image_mlp.ImageMLP_FashionMnist8_Relu32())

gradient_estimators = [
    grad_est_fn(quadratics.FixedDimQuadraticFamily(10)),
    grad_est_fn(mlp_task_family),
]

outer_trainer = gradient_learner.SingleMachineGradientLearner(
    lopt, gradient_estimators, theta_opt)
```

+++ {"id": "IKmWbqwhkxLp"}

To use this, we must first construct the initial state which contains a randomly initialized learned optimizer, as well as all the initial state of all the inner-problems (in this case, one set for the quadratics being trained, and the other from the MLP being trained).

```{code-cell}
---
executionInfo:
  elapsed: 2924
  status: ok
  timestamp: 1647562738598
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: N78WoigVkmUb
outputId: 9c275261-c03f-4de4-804f-b48efd1ea4ed
---
key = jax.random.PRNGKey(0)
outer_trainer_state = outer_trainer.init(key)
jax.tree_util.tree_map(lambda x: jnp.asarray(x).shape, outer_trainer_state)
```

+++ {"id": "bR6qSqckl6XG"}

This SingleMachineState contains the state of the `gradient_learner` (or the weights of the learned optimizer being trained as well as any variables needed to train these weights such as momentums (`theta_opt_state`).
It also contains the `gradient_estimator_states` which is a list containing the states of each gradient estimator.

We can train a single meta-step as follows. Note this could take a few minutes to compile.

```{code-cell}
---
executionInfo:
  elapsed: 10757
  status: ok
  timestamp: 1647562749465
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: 9lomKFrzkqkd
outputId: cbce9749-bf2e-4abd-d7bd-8822077d587d
---
next_state, loss, metrics = outer_trainer.update(
    outer_trainer_state, key, with_metrics=False)
```

+++ {"id": "PZNCltmgmfug"}

Now let's meta-train a few steps and watch the meta-loss go down.

```{code-cell}
---
executionInfo:
  elapsed: 38491
  status: ok
  timestamp: 1647562788069
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: 6q3_EN5qmle_
outputId: 234b28a7-8d3d-4740-8275-1d75ad7e54bc
---
losses = []
import tqdm

import os
# Pulling this from an environment variable so this file can be run in automated tests faster.
outer_train_steps = int(os.environ.get("LOPT_META_TRAIN_LENGTH", 500))

for i in tqdm.trange(outer_train_steps):
  outer_trainer_state, loss, metrics = outer_trainer.update(
      outer_trainer_state, key, with_metrics=False)
  losses.append(loss)
```

```{code-cell}
---
colab:
  height: 282
executionInfo:
  elapsed: 257
  status: ok
  timestamp: 1647562788441
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: cT_iJUTHmfWt
outputId: 436186c8-e7b5-4599-de10-aa7fc3200bc7
---
plt.plot(losses)
plt.xlabel("outer iteration")
plt.ylabel("outer loss")
```

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1647562788752
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: akK_PYCIkwDx
outputId: c3ad2258-b9ca-458d-fc52-58423d5dd0cc
---
metrics
```

+++ {"id": "kh27tduOwR3E"}

We see some oscillations in this loss value. This makes sense as we are meta-training with a truncated method. Depending on the location in the inner-problem the losses will be greater or smaller. With enough tasks this will average out, but often this is not required to obtain performant learned optimizers. This does mean, however, that evaluating a learned optimizer after the fact is important to obtain a realistic performance measurement.

+++ {"id": "gKImCte3n5gF"}

## Manual meta-gradient aggregation with applications to distributed training

Often when training a learned optimizer we seek to estimate meta-gradients over a wide variety of tasks spread across a number of different machines.
To do this we need to be explicit about what data needs to be sent from the central learner to compute updates (usually just the weights of the learned optimizer) and what data needs to be sent back (aggregated gradients + metrics and **not** the state of the unrolls for each inner-problem.)

To help manage this, we provide a class to manage the central learner's computation, and a function to compute updates which would run on each worker.


As this demo is in a colab, we will do everything in one process.
First, we will create the central learner. This is responsible for taking in gradients, updating the weights of the learned optimizer, and providing new data back to the workers.

```{code-cell}
:id: SbWqD3pbl3BW

theta_opt = opt_base.Adam(1e-3)
central_learner = gradient_learner.GradientLearner(lopt, theta_opt)
```

```{code-cell}
---
executionInfo:
  elapsed: 60
  status: ok
  timestamp: 1647562789178
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: SuKABLjk3D9R
outputId: de50cad0-aabd-4f71-9a0b-85cb20a94487
---
key = jax.random.PRNGKey(0)
central_state = central_learner.init(key)
jax.tree_util.tree_map(lambda x: jnp.asarray(x).shape, central_state)
```

+++ {"id": "YmbKXLcL3zDB"}

We can see here that this just contains the weights of the learned optimizer, plus the extra accumulators used by adam.

Next, we can compute gradient estimators, but first we must get the required state from the learner.

```{code-cell}
:id: 1MunNqVl4TSH

worker_weights = central_learner.get_state_for_worker(central_state)
```

+++ {"id": "GrBAGdre4bWI"}

Next, we can compute gradients on a given worker. As before we need to get a list of gradient estimators. We can use the same set we used before.

```{code-cell}
:id: D5EfexNg5H6p

max_length = 300
trunc_sched = truncation_schedule.LogUniformLengthSchedule(
    min_length=100, max_length=max_length)


def grad_est_fn(task_family):
  trunc_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family,
      lopt,
      trunc_sched,
      num_tasks=16,
      random_initial_iteration_offset=max_length)
  return truncated_pes.TruncatedPES(trunc_step, trunc_length=50)


mlp_task_family = tasks_base.single_task_to_family(
    image_mlp.ImageMLP_FashionMnist8_Relu32())

gradient_estimators = [
    grad_est_fn(quadratics.FixedDimQuadraticFamily(10)),
    grad_est_fn(mlp_task_family)
]
```

+++ {"id": "tmJEzRjw5LSA"}

Next, we need to kick things off by first computing the initial states for each of the gradient estimators.

```{code-cell}
---
executionInfo:
  elapsed: 1143
  status: ok
  timestamp: 1647562879797
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: gsPQvD4Y5RQd
outputId: ee93ac90-fed8-42d5-e376-2aa52b7da2df
---
unroll_states = [
    grad.init_worker_state(worker_weights, key=jax.random.fold_in(key, i))
    for (i, grad) in enumerate(gradient_estimators)
]
```

+++ {"id": "gE4nf01r6AcB"}

Next we can use these states to estimate a meta-gradient!

```{code-cell}
---
executionInfo:
  elapsed: 10255
  status: ok
  timestamp: 1647562891517
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: sd_DuFzS3Gr8
outputId: b742898c-ecb0-4f43-c9e8-19ac8a484770
---
out = gradient_learner.gradient_worker_compute(
    worker_weights,
    gradient_estimators,
    unroll_states,
    key=key,
    with_metrics=False)
```

+++ {"id": "B01vLh7V5-zu"}

This produces a couple of different outputs bundled together in a dataclass.

```{code-cell}
---
executionInfo:
  elapsed: 83
  status: ok
  timestamp: 1647562891712
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: Ax495URy6LSA
outputId: d66c3912-719a-4867-c7b4-35ccbf3697b7
---
[x for x in dir(out) if not x.startswith("__") and x != "replace"]
```

+++ {"id": "HI-xgDib0c6d"}

Most importantly we have `to_put` which contains information that should be sent to the central learner, and `unroll_states` which contains the next unroll states.

+++ {"id": "VAKMmIRt6e60"}

Now with more than one worker, we would pass back a list of these gradients. In this demo, we will just use a single one, and pass this directly into the central learner to get the next meta-iteration. With more workers, this would contain a different gradient estimator from each worker.

```{code-cell}
:id: pne3-Yy147tW

key1, key = jax.random.split(key)
grads_list = [out.to_put]
central_state, metrics = central_learner.update(central_state, grads_list, key=key1)
```

+++ {"id": "i0H_re9cVW1b"}

And we can do this over and over again. This time let's do it with more than one gradient estimate.

```{code-cell}
---
executionInfo:
  elapsed: 62457
  status: ok
  timestamp: 1647562954574
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: KNIUGUvKVwld
outputId: 997ad5f5-e7a1-4b4c-bdc5-02f322178273
---
losses = []

outer_train_steps = int(os.environ.get("LOPT_META_TRAIN_LENGTH", 500))

for i in tqdm.trange(outer_train_steps):
  worker_weights = central_learner.get_state_for_worker(central_state)

  key1, key = jax.random.split(key)
  out = gradient_learner.gradient_worker_compute(
      worker_weights,
      gradient_estimators,
      unroll_states,
      key=key1,
      with_metrics=False)
  # extract the next unroll state output for the next iteration.
  unroll_states = out.unroll_states

  key1, key = jax.random.split(key)
  central_state, metrics = central_learner.update(
      central_state, [out.to_put], key=key1)
  losses.append(out.to_put.mean_loss)
```

```{code-cell}
---
colab:
  height: 296
executionInfo:
  elapsed: 177
  status: ok
  timestamp: 1647563002848
  user:
    displayName: ''
    photoUrl: ''
    userId: ''
  user_tz: 240
id: WszEeLOBJw0x
outputId: d548cd42-aa08-407b-e65c-a2e3bacb7b91
---
plt.plot(losses)
plt.xlabel("outer iteration")
plt.ylabel("outer loss")
```

```{code-cell}
:id: nyqYdAFIaFJh


```
