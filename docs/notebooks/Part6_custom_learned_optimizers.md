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

+++ {"id": "uSUkKaMchXQ9"}

# Part 6: Custom learned optimizer architectures
In [Part 1](https://learned-optimization.readthedocs.io/en/latest/notebooks/Part1_Introduction.html) we introduced the `LearnedOptimizer` abstraction. In this notebook we will discuss how to construct one. We will show 3 examples: Meta-learning hyper parameters, a per-parameter optimizer, and a hyper parameter controller.

```{code-cell}
---
executionInfo:
  elapsed: 58
  status: ok
  timestamp: 1644472716995
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: LxTj6OcNLswq
---
import flax
from typing import Any
import jax.numpy as jnp
import jax

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
```

+++ {"id": "0fgVMtdVLuC0"}

## Meta-Learnable hyper parameters
Let's first start by defining a learned optimizer with meta-learned hyper parameters. For this, we will choose SGD as the base optimizer, and meta-learn a learning rate and weight decay.


First, we define the state of the learned optimizer. This state is used to keep track of the learned optimizer weights. It contains the inner parameters (`params`), the inner `model_state` which is None unless there are non-gradient updated parameters in the inner problem (such as batchnorm statistics), and `iteration` which contains the inner-training step.

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1644472718443
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: tmkJTQNSLvjj
---
@flax.struct.dataclass
class LOptState:
  params: Any
  model_state: Any
  iteration: jnp.ndarray
```

+++ {"id": "nSO2PgeqMF3X"}

Next for the main optimizer.
See the comments inline the code description.

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1644472737722
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: feQ6ZWlmNUWI
---
MetaParams = Any  # typing definition to label some types below

class MetaSGDWD(lopt_base.LearnedOptimizer):
  def __init__(self, initial_lr=1e-3, initial_wd=1e-2):
    self._initial_lr = initial_lr
    self._initial_wd = initial_wd

  def init(self, key) -> MetaParams:
    """Initialize the weights of the learned optimizer.

    In this case the initial learning rate, and initial weight decay.
    """
    # These are the initial values with which we would start meta-optimizing from
    return {
        "log_lr": jnp.log(self._initial_lr),
        "log_wd": jnp.log(self._initial_wd)
    }

  def opt_fn(self, theta: MetaParams) -> opt_base.Optimizer:
    # define an anonymous class which implements the optimizer.
    # this captures over the meta-parameters, theta.

    class _Opt(opt_base.Optimizer):
      def init(self, params, model_state=None, **kwargs) -> LOptState:
        # For our inital inner-opt state we pack the params, model state,
        # and iteration into the LOptState dataclass.
        return LOptState(
            params=params,
            model_state=model_state,
            iteration=jnp.asarray(0, dtype=jnp.int32))

      def update(self,
                 opt_state: LOptState,
                 grads,
                 model_state=None,
                 **kwargs) -> LOptState:
        """Perform the actual update."""
        # We grab the meta-parameters and transform them back to their original
        # space
        lr = jnp.exp(theta["log_lr"])
        wd = jnp.exp(theta["log_wd"])

        # Next we define the weight update.
        def _update_one(p, g):
          return p - g * lr - p * wd

        next_params = jax.tree_util.tree_map(_update_one, opt_state.params, grads)
        # Pack the new parameters back up
        return LOptState(
            params=next_params,
            model_state=model_state,
            iteration=opt_state.iteration + 1)
    return _Opt()
```

+++ {"id": "KIJ2gyMBNpi4"}

To test this out, we can feed in a fake set of params and gradients and look at the new parameter values.

```{code-cell}
---
executionInfo:
  elapsed: 129
  status: ok
  timestamp: 1644473371088
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: BB9kc27fNpAB
outputId: dcb51eea-4a4d-46e9-a9a2-8563799b797b
---
lopt = MetaSGDWD()
key = jax.random.PRNGKey(0)
theta = lopt.init(key)
opt = lopt.opt_fn(theta)
fake_params = {"a": 1.0, "b": 2.0}
opt_state = opt.init(fake_params)
fake_grads = {"a": -1.0, "b": 1.0}
new_opt_state = opt.update(opt_state, fake_grads)

opt.get_params(new_opt_state)
```

+++ {"id": "xzGRP13ZN4rJ"}

## Per Parameter learned optimizer
Per parameter learned optimizers involves computing some learned function on
each parameter of the inner-model. Because these calculations are done on
every parameter, the computational cost of applying the optimizer grows linearly
with the number of parameters in the inner problem.

To demonstrate this kind of optimizer, we implement a small MLP which operates on gradients,
momentum values, and parameters and produces a scalar update.
This MLP is applied to each parameter independently. As such, it takes in three
scalar inputs (the gradient, momentum, and parameter value), and produces two
outputs which are combined to form a single scalar.
The same MLP is then applied to every weight.

```{code-cell}
---
executionInfo:
  elapsed: 60
  status: ok
  timestamp: 1644473371311
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: WZY4cds2PUA1
---
@flax.struct.dataclass
class PerParamState:
  params: Any
  model_state: Any
  iteration: jnp.ndarray
  momentums: Any
```

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1644473578782
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: RUCGENb7N6D9
---
import haiku as hk

class PerParamMLP(lopt_base.LearnedOptimizer):
  def __init__(self, decay=0.9, hidden_size=64):
    self.decay = decay
    self.hidden_size = hidden_size

    def forward(grads, momentum, params):
      features = jnp.asarray([params, momentum, grads])
      # transpose to have features dim last. The MLP will operate on this,
      # and treat the leading dimensions as a batch dimension.
      features = jnp.transpose(features,
                               list(range(1, 1 + len(grads.shape))) + [0])

      outs = hk.nets.MLP([self.hidden_size, 2])(features)

      scale = outs[..., 0]
      mag = outs[..., 1]
      # Compute a step as follows.
      return scale * 0.01 * jnp.exp(mag * 0.01)

    self.net = hk.without_apply_rng(hk.transform(forward))



  def init(self, key) -> MetaParams:
    """Initialize the weights of the learned optimizer."""
    # to initialize our neural network, we must pass in a batch that looks like
    # data we might train on.
    # Because we are operating per parameter, the shape of this batch doesn't
    # matter.
    fake_grads = fake_params = fake_mom = jnp.zeros([10, 10])
    return {"nn": self.net.init(key, fake_grads, fake_mom, fake_params)}

  def opt_fn(self, theta: MetaParams) -> opt_base.Optimizer:
    # define an anonymous class which implements the optimizer.
    # this captures over the meta-parameters, theta.

    parent = self

    class _Opt(opt_base.Optimizer):
      def init(self, params, model_state=None, **kwargs) -> LOptState:
        # In addition to params, model state, and iteration, we also need the
        # initial momentum values.

        momentums = jax.tree_util.tree_map(jnp.zeros_like, params)

        return PerParamState(
            params=params,
            model_state=model_state,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            momentums=momentums)

      def update(self,
                 opt_state: LOptState,
                 grads,
                 model_state=None,
                 **kwargs) -> LOptState:
        """Perform the actual update."""

        # update all the momentums
        def _update_one_momentum(m, g):
          return m * parent.decay + (g * (1 - parent.decay))

        next_moms = jax.tree_util.tree_map(_update_one_momentum, opt_state.momentums,
                                 grads)

        # Update all the params
        def _update_one(g, m, p):
          step = parent.net.apply(theta["nn"], g, m, p)
          return p - step

        next_params = jax.tree_util.tree_map(_update_one, opt_state.params, grads,
                                   next_moms)

        # Pack the new parameters back up
        return PerParamState(
            params=next_params,
            model_state=model_state,
            iteration=opt_state.iteration + 1,
            momentums=next_moms)
    return _Opt()
```

+++ {"id": "EjiNPZSnQ4Ab"}

Now let's look at what these meta-parameters look like.

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1644473615597
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: 0FFWo-xjQFJu
outputId: 7f736eb0-d3a5-4d25-c6e0-4dcde27f406c
---
lopt = PerParamMLP()
key = jax.random.PRNGKey(0)
theta = lopt.init(key)
jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), theta)
```

+++ {"id": "NSywvQWJRCIU"}

We have a 2 layer MLP. The first layer has 3 input channels (for grads, momentum, parameters), into 64 (hidden size), into 2 for output.

We can again apply our optimizer.

```{code-cell}
---
executionInfo:
  elapsed: 1
  status: ok
  timestamp: 1644473678355
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: 9oNudcVgQ72x
---
opt = lopt.opt_fn(theta)
fake_params = {"a": jnp.ones([2, 3]), "b": jnp.ones([1])}
opt_state = opt.init(fake_params)
fake_grads = {"a": -jnp.ones([2, 3]), "b": -jnp.ones([1])}
new_opt_state = opt.update(opt_state, fake_grads)
```

+++ {"id": "wKCBWC6gRQPM"}

We can see both params, and momentum was updated.

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1644473698434
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: NBVFeyJRRP3X
outputId: 3712a0fd-d3f2-49a7-da6f-47f5bdc33f35
---
print(opt.get_params(new_opt_state))
print(new_opt_state.momentums)
```

+++ {"id": "WB7LuabVRim_"}

## Meta-learned RNN Controllers

Another kind of learned optimizer architecture consists of a recurrent "controller" which modifies and sets the hyper parameters of some base model.
These optimizers often have low overhead as computing hparams to use is often much cheaper than computing the underlying gradients. These optimizers also don't require complex computations to be done at each parameter like the per parameter optimizers above.

To demonstrate this family, we will implement an adaptive learning rate optimizer.

The RNN we will use needs to operate on some set of features and outputs. For simplicity our learned optimizer will just use the loss as a feature, and produces a learning rate.
Because it is a recurrent model, we must also take in the previous and next RNN state. This loss is NOT provided into all optimizers and thus some care should be taken -- anything using this optimizer must know about the loss.


For this RNN, we use haiku for no particularly strong reason (Flax, or any other neural network library which allows for creating purely functional NN would work.)

This optimizer will additionally have a meta-learnable initial RNN State. We desire this state to be meta-learned and thus it must be constructed by `LearnedOptimizer.init`. This state needs to be updated while applying the optimizer, so when we construct the inner-optimizer state.

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1644474159670
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: Y5qEGiPGTCAP
---
@flax.struct.dataclass
class HParamControllerInnerOptState:
  params: Any
  model_state: Any
  iteration: Any
  rnn_hidden_state: Any
```

+++ {"id": "A81lpZa7TXdc"}

First we will define some helper functions which perform the compute of the learned optimizer.

```{code-cell}
:id: VWjvA9AETb94

import haiku as hk

def rnn_mod():
  return hk.LSTM(128)

@hk.transform
def initial_state_fn():
  rnn_hidden_state = rnn_mod().initial_state(batch_size=1)
  return rnn_hidden_state

@hk.transform
def forward_fn(hidden_state, input):
  mod = rnn_mod()
  output, next_state = mod(input, hidden_state)
  log_lr = hk.Linear(1)(output)
  return next_state, jnp.exp(log_lr) * 0.01
```

+++ {"id": "sKVF8_UhTgH6"}

Now for the full optimizer

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1644474352955
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: CNNjLi7Dm7Wz
---
class HParamControllerLOPT(lopt_base.LearnedOptimizer):
  def init(self, key):
    """Initialize weights of learned optimizer."""
    # Only one input -- just the loss.
    n_input_features = 1
    # This takes no input parameters -- hence the {}.
    initial_state = initial_state_fn.apply({}, key)

    fake_input_data = jnp.zeros([1, n_input_features])
    rnn_params = forward_fn.init(key, initial_state, fake_input_data)
    return {"rnn_params": rnn_params, "initial_rnn_hidden_state": initial_state}

  def opt_fn(self, theta):
    class _Opt(opt_base.Optimizer):
      def init(self, params, model_state=None, **kwargs):
        # Copy the initial, meta-learned rnn state into the inner-parameters
        # so that it can be updated by the RNN.
        return HParamControllerInnerOptState(
            params=params,
            model_state=model_state,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            rnn_hidden_state=theta["initial_rnn_hidden_state"])

      def update(self, opt_state, grads, loss=None, model_state=None, **kwargs):
        # As this loss is not part of the default Optimizer definition, we assert
        # that it is non None
        assert loss is not None

        # Add a batch dimension to the loss
        batched_loss = jnp.reshape(loss, [1, 1])

        # run the RNN
        rnn_forward = hk.without_apply_rng(forward_fn).apply
        next_rnn_state, lr = rnn_forward(theta["rnn_params"],
                                         opt_state.rnn_hidden_state,
                                         batched_loss)

        # use the results of the RNN to update the parameters.
        def update_one(p, g):
          return p - g * lr

        next_params = jax.tree_util.tree_map(update_one, opt_state.params, grads)

        return HParamControllerInnerOptState(
            params=next_params,
            model_state=model_state,
            iteration=opt_state.iteration + 1,
            rnn_hidden_state=next_rnn_state)

    return _Opt()
```

+++ {"id": "mVjY1-cHT01N"}

We can apply this optimizer on some fake parameters. If we look at the state, we will see the parameter values, as well as the rnn hidden state.

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1644474625301
  user:
    displayName: Luke Metz
    photoUrl: https://lh3.googleusercontent.com/a-/AOh14Gif9m36RuSe53tMVslYQLofCkRX0_Y47HVoDh3u=s64
    userId: 07706439306199750899
  user_tz: 480
id: Bei372EpT1MY
outputId: 6807d2ba-07b1-4ad9-d53c-cf5a4cbc49e1
---
lopt = HParamControllerLOPT()
theta = lopt.init(key)
opt = lopt.opt_fn(theta)

params = {"a": jnp.ones([3, 2]), "b": jnp.ones([2, 1])}
opt_state = opt.init(params)
fake_grads = {"a": -jnp.ones([3, 2]), "b": -jnp.ones([2, 1])}
opt_state = opt.update(opt_state, fake_grads, loss=1.0)
jax.tree_util.tree_map(lambda x: x.shape, opt_state)
```

+++ {"id": "RLAJP-MdU_5I"}

## More LearnedOptimizer architectures

Many more learned optimizer architectures are implemented inside the [learned_optimization/learned_optimizers](https://github.com/google/learned_optimization/tree/main/learned_optimization/learned_optimizers) folder. These include:

* `nn_adam`: which implements a more sophisticated hyper parameter controller which controls Adam hparams.

* `mlp_lopt` and `adafac_mlp_lopt`: which implement more sophisticated per-parameter learned optimizers.

* `rnn_mlp_opt`: Implements a hierarchical learned optimizer. A per tensor RNN is used to compute hidden state which is passed to a per-parameter MLP which does the actual weight updates.
