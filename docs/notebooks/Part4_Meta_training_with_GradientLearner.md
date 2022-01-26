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

+++ {"id": "a7QBk71UMifh"}

# Part 4: Meta-training with GradientLearner

```{code-cell}
---
executionInfo:
  elapsed: 118
  status: ok
  timestamp: 1642474445264
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: MimfK6lp0vq9
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
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1642474445452
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: txx2_SmH8NhL
---
theta_opt = opt_base.Adam(1e-3)

lopt = mlp_lopt.MLPLOpt()
max_length = 300
trunc_sched = truncation_schedule.LogUniformLengthSchedule(
    min_length=100, max_length=max_length)


def grad_est_fn(task_family):
  return truncated_pes.TruncatedPES(
      task_family=task_family,
      learned_opt=lopt,
      trunc_sched=trunc_sched,
      num_tasks=4,
      trunc_length=50,
      random_initial_iteration_offset=max_length)


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
  elapsed: 1877
  status: ok
  timestamp: 1642474447458
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: N78WoigVkmUb
outputId: 449c35c3-462f-42a8-ec77-7a66529a82cb
---
key = jax.random.PRNGKey(0)
outer_trainer_state = outer_trainer.init(key)
jax.tree_map(lambda x: jnp.asarray(x).shape, outer_trainer_state)
```

+++ {"id": "bR6qSqckl6XG"}

This SingleMachineState contains the state of the `gradient_learner` (or the weights of the learned optimizer being trained as well as any variables needed to train these weights such as momentums (`theta_opt_state`).
It also contains the `gradient_estimator_states` which is a list containing the states of each gradient estimator.

We can train a single meta-step as follows. Note this could take a few minutes to compile.

```{code-cell}
---
executionInfo:
  elapsed: 15796
  status: ok
  timestamp: 1642474463407
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: 9lomKFrzkqkd
outputId: ee8eaff5-7920-49a0-bcf3-bfffbd752c93
---
next_state, loss, metrics = outer_trainer.update(
    outer_trainer_state, key, with_metrics=False)
```

+++ {"id": "PZNCltmgmfug"}

Now let's meta-train a few steps and watch the meta-loss go down.

```{code-cell}
---
executionInfo:
  elapsed: 31861
  status: ok
  timestamp: 1642474495485
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: 6q3_EN5qmle_
outputId: 9e25f1b8-b2ec-4a60-e22d-ee3acc99c249
---
losses = []
import tqdm

import os
# Pulling this from an environment variable so this file can be tested.
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
  elapsed: 120
  status: ok
  timestamp: 1642474495753
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: cT_iJUTHmfWt
outputId: 75cb6800-27c4-481c-938e-fd1b90f7e73e
---
plt.plot(losses)
```

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1642474496032
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: akK_PYCIkwDx
outputId: f86e1542-3e87-4af0-8379-cdf178b40caf
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
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1642475145999
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: SbWqD3pbl3BW
---
theta_opt = opt_base.Adam(1e-3)
central_learner = gradient_learner.GradientLearner(lopt, theta_opt)
```

```{code-cell}
---
executionInfo:
  elapsed: 84
  status: ok
  timestamp: 1642475146666
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: SuKABLjk3D9R
outputId: c69694a8-11fd-4112-91a4-b0711f1efc7b
---
key = jax.random.PRNGKey(0)
central_state = central_learner.init(key)
jax.tree_map(lambda x: jnp.asarray(x).shape, central_state)
```

+++ {"id": "YmbKXLcL3zDB"}

We can see here that this just contains the weights of the learned optimizer, plus the extra accumulators used by adam.

Next, we can compute gradient estimators, but first we must get the required state from the learner.

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1642475147260
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: 1MunNqVl4TSH
---
worker_weights = central_learner.get_state_for_worker(central_state)
```

+++ {"id": "GrBAGdre4bWI"}

Next, we can compute gradients on a given worker. As before we need to get a list of gradient estimators. We can use the same set we used before.

```{code-cell}
---
executionInfo:
  elapsed: 140
  status: ok
  timestamp: 1642475147690
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: D5EfexNg5H6p
---
max_length = 300
trunc_sched = truncation_schedule.LogUniformLengthSchedule(
    min_length=100, max_length=max_length)


def grad_est_fn(task_family):
  return truncated_pes.TruncatedPES(
      task_family=task_family,
      learned_opt=lopt,
      trunc_sched=trunc_sched,
      num_tasks=16,
      trunc_length=50,
      random_initial_iteration_offset=max_length)


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
  elapsed: 2654
  status: ok
  timestamp: 1642475150735
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: gsPQvD4Y5RQd
outputId: f99121fc-f543-427d-e882-d21819330d75
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
  elapsed: 18204
  status: ok
  timestamp: 1642475169126
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: sd_DuFzS3Gr8
outputId: fb768953-cdcb-4a30-d4fc-8b3243526a5d
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
  elapsed: 3
  status: ok
  timestamp: 1642475169475
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: Ax495URy6LSA
outputId: cd669daf-5d28-4868-c046-f0b6d2b0af86
---
[x for x in dir(out) if not x.startswith("__") and x != "replace"]
```

+++ {"id": "HI-xgDib0c6d"}

Most importantly we have `to_put` which contains information that should be sent to the central learner, and `unroll_states` which contains the next unroll states.

+++ {"id": "VAKMmIRt6e60"}

Now with more than one worker, we would pass back a list of these gradients. In this demo, we will just use a single one, and pass this directly into the central learner to get the next meta-iteration. With more workers, this would contain a different gradient estimator from each worker.

```{code-cell}
---
executionInfo:
  elapsed: 185
  status: ok
  timestamp: 1642475170013
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: pne3-Yy147tW
---
grads_list = [out.to_put]
central_state, metrics = central_learner.update(central_state, grads_list)
```

+++ {"id": "i0H_re9cVW1b"}

And we can do this over and over again. This time let's do it with more than one gradient estimate.

```{code-cell}
---
executionInfo:
  elapsed: 62042
  status: ok
  timestamp: 1642475232310
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: KNIUGUvKVwld
outputId: 37709a54-0c73-4210-9eb1-54dbe9fa0575
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

  central_state, metrics = central_learner.update(central_state, [out.to_put])
  losses.append(out.to_put.mean_loss)
```

```{code-cell}
---
colab:
  height: 282
executionInfo:
  elapsed: 227
  status: ok
  timestamp: 1642475232707
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: WszEeLOBJw0x
outputId: 64a2a6f2-9032-40d9-c1fc-5af448bcedde
---
plt.plot(losses)
```
