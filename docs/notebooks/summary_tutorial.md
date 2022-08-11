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

+++ {"id": "ryqPvTKI19zH"}

# Summary tutorial: Getting metrics out of your models

The goal of the `learned_optimization.summary` module is to seamlessly allow researchers to annotate and extract data from *within* a jax computation / machine learning model.
This could be anything from mean and standard deviation of an activation, to looking at the distribution of outputs.

Doing this in Jax can be challenging at times as code written in Jax can make use of a large number of function transformations making it difficult to reach in and look at a value.

This notebook discusses the `learned_optimization.summary` module which provides one solution to this problem and is discussed in this notebook.

+++ {"id": "aJfX-Mda59NC"}

## Deps

+++ {"id": "NGqJCT0MwlvI"}

In addition to `learned_optimization`, the summary module requires `oryx`. This can be a bit finicky to install at the moment as it relies upon particular versions of tensorflow (even though we never use these pieces).
All of `learned_optimization` will run without `oryx`, but to get summaries this module must be installed.
In a colab this is even more annoying as we must first upgrade the versions of some installed modules, restart the colab kernel, and then proceed to run the remainder of the cells.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: dLD5VE3EACSI
outputId: fe7f4bea-c9d1-4b0c-91e5-75d61a60a11f
---
!pip install --upgrade git+https://github.com/google/learned_optimization.git oryx tensorflow==2.8.0rc0 numpy
```

+++ {"id": "gzqk9f_t6f8J"}

To check that everything is operating as expected we can check that the imports succeed.

```{code-cell}
:id: kbWUpquq6XSu

import oryx
from learned_optimization import summary
assert summary.ORYX_LOGGING
```

+++ {"id": "Lz6MTETQ4R11"}

## Basic Example
Let's say that we have the following function and we wish to look at the `to_look_at` value.

```{code-cell}
:id: Jugy6h9d4QT0

import jax
import jax.numpy as jnp
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: ozBppGEc4QT0
outputId: 2b9b497f-5cf7-45e1-dd6f-0195e13b7a24
---
def forward(params):
  to_look_at = jnp.mean(params) * 2.
  return params


def loss(parameters):
  loss = jnp.mean(forward(parameters)**2)
  return loss


value_grad_fn = jax.jit(jax.value_and_grad(loss))
value_grad_fn(1.0)
```

+++ {"id": "9WzMuzf44j0L"}

With the summary module we can first annotate this value with `summary.summary`

```{code-cell}
:id: K7N39btX4jUe

def forward(params):
  to_look_at = jnp.mean(params) * 2.
  summary.summary("to_look_at", to_look_at)
  return params


@jax.jit
def loss(parameters):
  loss = jnp.mean(forward(parameters)**2)
  return loss
```

+++ {"id": "AL9_xgfR4yPS"}

Then we can transform the `loss` function with the function transformation: `summary.with_summary_output_reduced`.
This transformation goes through the computation and extracts all the tagged values and returns them to us by name in a dictionary.
In implementation, all the hard work here is done by the wonderful `oryx` library (in particular [harvest](https://github.com/tensorflow/probability/blob/main/spinoffs/oryx/oryx/core/interpreters/harvest.py)).
When we wrap a function this, we return a tuple containing the original result, and a dictionary with the desired metrics.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: hZQkB6Um8PI5
outputId: 984e4f64-7562-48ae-ca68-ff4014037553
---
result, metrics = summary.with_summary_output_reduced(loss)(1.)
result, metrics
```

+++ {"id": "EJ7sPabc-zby"}

As currently returned, the dictionary contains extra information as well as potentially duplicate values. We can collapse these metrics into a single value with the following:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: ZpZxTgsF-zJv
outputId: 920de601-0c5e-4342-8bde-cf5297dc4635
---
summary.aggregate_metric_list([metrics])
```

+++ {"id": "tPZ-YM6I_Dhk"}

The keys of this dictionary first show how the metric was aggregated. In this case there is only a single metric so the aggregation is ignored. This is followed by `||` and then the summary name.

+++ {"id": "Pd9RDqbVAB-L"}

One benefit of this function transformation is it can be nested with other jax function transformations. For example we can jit the transformed function like so:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 3f7vaGON_7Dc
outputId: 7ad7c330-a0eb-4df7-96c2-0ae6d92ac252
---
result, metrics = jax.jit(summary.with_summary_output_reduced(loss))(1.)
summary.aggregate_metric_list([metrics])
```

+++ {"id": "ZeriYYoTJvKd"}

At this point `aggregate_metric_list` cannot be jit. In practice this is fine as it performs very little computation.

+++ {"id": "FfONK6la_QYr"}

## Aggregation of the same name summaries.

Consider the following fake function which calls a `layer` function twice.
This `layer` function creates a summary and thus two summary are created.
When running the transformed function we see not one, but two values returned.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: SDToo5ZH-8kP
outputId: 1a6956f1-7968-4e36-e5bf-de05334ca58e
---
def layer(params):
  to_look_at = jnp.mean(params) * 2.
  summary.summary("to_look_at", to_look_at)
  return params * 2


@jax.jit
def loss(parameters):
  loss = jnp.mean(layer(layer(parameters))**2)
  return loss


result, metrics = summary.with_summary_output_reduced(loss)(1.)
result, metrics
```

+++ {"id": "JZsHOz6J_t3-"}

These values can be combined with `aggregate_metric_list` as before, but this time the aggregation takes the mean. This `mean` is specified by the `aggregation` keyword argument in `summary.summary` which defaults to `mean`.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: TW4wQ-kB-cCQ
outputId: 824fd411-5cad-4368-9698-d1a04e309981
---
summary.aggregate_metric_list([metrics])
```

+++ {"id": "ZXtFMhmqANGJ"}

Another useful aggregation mode is "sample". As jax's random numbers are stateless, an additional RNG key must be passed in for this to work.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: hvfgKVZhAeRW
outputId: fc44541a-82b5-4a2d-f103-e3b8f929aa56
---
def layer(params):
  to_look_at = jnp.mean(params) * 2.
  summary.summary("to_look_at", to_look_at, aggregation="sample")
  return params * 2


@jax.jit
def loss(parameters):
  loss = jnp.mean(layer(layer(parameters))**2)
  return loss


key = jax.random.PRNGKey(0)
result, metrics = summary.with_summary_output_reduced(loss)(
    1., sample_rng_key=key)
summary.aggregate_metric_list([metrics])
```

+++ {"id": "0Q98mny6AyNH"}

Finally, there is `"collect"` which concatenates all the values together into one long tensor after first raveling all the inputs. This is useful for extracting distributions of quantities.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: SzVbsqZIA2O9
outputId: 9fec7c59-f8bf-4aa1-fa67-3e4bc75153d2
---
def layer(params):
  to_look_at = jnp.mean(params) * 2.
  summary.summary(
      "to_look_at", jnp.arange(10) * to_look_at, aggregation="collect")
  return params * 2


@jax.jit
def loss(parameters):
  loss = jnp.mean(layer(layer(parameters))**2)
  return loss


key = jax.random.PRNGKey(0)
result, metrics = summary.with_summary_output_reduced(loss)(
    1., sample_rng_key=key)
summary.aggregate_metric_list([metrics])
```

+++ {"id": "nCN3ET9IQd7A"}

## Summary Scope
Sometimes it is useful to be able to group all summaries from a function inside some code block with some common name. This can be done with the `summary_scope` context.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 1SM-Ab-CQgWw
outputId: b9dbd717-006d-465e-c4af-b10445b08d2c
---
@jax.jit
def loss(parameters):
  with summary.summary_scope("scope1"):
    summary.summary("to_look_at", parameters)

  with summary.summary_scope("nested"):
    summary.summary("summary2", parameters)

    with summary.summary_scope("scope2"):
      summary.summary("to_look_at", parameters)
  return parameters


key = jax.random.PRNGKey(0)
result, metrics = summary.with_summary_output_reduced(loss)(
    1., sample_rng_key=key)
summary.aggregate_metric_list([metrics])
```

```{code-cell}
:id: pOEJJw2EQd2h


```

+++ {"id": "4pzFDb3FBfi-"}

## Usage with function transforms.
Thanks to `oryx`, all of this functionality works well across a variety of function transformations.
Here is an example with a scan, vmap, and jit.
The aggregation modes will aggregate across all timesteps, and all batched dimensions.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: FFW4l5RPBn7I
outputId: d621048f-921c-4d40-f6ae-9b516c12078a
---
@jax.jit
def fn(a):
  summary.summary("other_val", a[2])

  def update(state, _):
    s = state + 1
    summary.summary("mean_loop", s[0])
    summary.summary("collect_loop", s[0], aggregation="collect")
    return s, s

  a, _ = jax.lax.scan(update, a, jnp.arange(20))
  return a * 2


vmap_fn = jax.vmap(fn)

result, metrics = jax.jit(summary.with_summary_output_reduced(vmap_fn))(
    jnp.tile(jnp.arange(4), (2, 2)))
summary.aggregate_metric_list([metrics])
```

+++ {"id": "gwpKu6neCjyN"}

## Optionally compute metrics: `@add_with_metrics`

Oftentimes it is useful to define two "versions" of a function -- one with metrics, and one without -- as sometimes the computation of the metrics adds unneeded overhead that does not need to be run every iteration.
To create these two versions one can simply wrap the function with the `add_with_summary` decorator.
This adds both a keyword argument, and an extra return to the wrapped function which switches between computing metrics, or not.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Nlw9j1NUCuXD
outputId: 421f5b94-5446-4fd9-d2aa-f944c54e4a97
---
from learned_optimization import summary
import functools


def layer(params):
  to_look_at = jnp.mean(params) * 2.
  summary.summary("to_look_at", jnp.arange(10) * to_look_at)
  return params * 2


@functools.partial(jax.jit, static_argnames="with_summary")
@summary.add_with_summary
def loss(parameters):
  loss = jnp.mean(layer(layer(parameters))**2)
  return loss


res, metrics = loss(1., with_summary=False)
print("No metrics", summary.aggregate_metric_list([metrics]))

res, metrics = loss(1., with_summary=True)
print("With metrics", summary.aggregate_metric_list([metrics]))
```

+++ {"id": "SnRKsVLbHA1t"}

## Limitations and Gotchas

+++ {"id": "LHCCoTCbLtpD"}

### Requires value traced to be a function of input

At the moment summary.summary MUST be called with a descendant of whatever input is passed into the function wrapped by `with_summary_output_reduced`. In practice this is almost always the case as we seek to monitor changing values rather than constants.

To demonstrate this, note how the constant value is NOT logged out, but if add it to `a*0` it does become logged out.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: iIbrjrJ4HEd-
outputId: b1ccd1db-5615-45b9-b52c-0ae78ea9369f
---
def monitor(a):
  summary.summary("with_input", a)
  summary.summary("constant", 2.0)
  summary.summary("constant_with_inp", 2.0 + (a * 0))
  return a


result, metrics = summary.with_summary_output_reduced(monitor)(1.)
summary.aggregate_metric_list([metrics])
```

+++ {"id": "yvEW6XfTH5_N"}

The rational for why this is a bit of a rabbit hole, but it is related to how tracing in jax work and is beyond the scope of this notebook.

+++ {"id": "wk9gKeH7LvHb"}

### No support for jax.lax.cond

At this point one cannot extract summaries out of jax conditionals. Sorry. If this is a limitation to you let us know as we have some ideas to make this work.

+++ {"id": "xMUpBsUUK4un"}

### No dynamic names

At the moment, the tag, or the name of the summary, must be a string known at compile time. There is no support for dynamic summary names.

+++ {"id": "6s4B2eUoz7nV"}

## Alternatives for extracting information
Using this module is not the only way to extract information from a model. We discuss a couple other approaches.

+++ {"id": "ZoekEh4J1geQ"}

### "Thread" metrics through
One way to extract data from a function is to simply return the things we want to look at. As functions become more complex and nested this can become quite a pain as each one of these functions must pass out metric values. This process of spreading data throughout a bunch of functions is called "threading".

Threading also requires all pieces of code to be involved -- as one must thread these metrics everywhere.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: q78BOx1X1uiN
outputId: 4bb8807f-4345-4e8a-c231-9bb482222352
---
def lossb(p):
  to_look_at = jnp.mean(123.)
  return p * 2, to_look_at


def loss(parameters):
  l = jnp.mean(parameters**2)
  l, to_look_at = lossb(l)
  return l, to_look_at


value_grad_fn = jax.jit(jax.value_and_grad(loss, has_aux=True))
(loss, to_look_at), g = value_grad_fn(1.0)
print(to_look_at)
```

+++ {"id": "jNt9CNJf2HJN"}

### jax.experimental.host_callback

Jax has some support to send data back from an accelerator back to the host while a ja program is running. This is exposed in jax.experimental.host_callback.

One can use this to print which is a quick way to get data out of a network.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 1Ih2LxP22MZD
outputId: 0dd0b8ec-2c9e-414d-eadf-843122b7b8ab
---
from jax.experimental import host_callback as hcb


def loss(parameters):
  loss = jnp.mean(parameters**2)
  to_look_at = jnp.mean(123.)
  hcb.id_print(to_look_at, name="to_look_at")
  return loss


value_grad_fn = jax.jit(jax.value_and_grad(loss))
_ = value_grad_fn(1.0)
```

+++ {"id": "06Dvp3xrEA-R"}

It is also possible to extract data out with [host_callback.id_tap](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html#jax.experimental.host_callback.id_tap). We experimented with this briefly for a summary library but found both performance issues and increased complexity around custom transforms.

```{code-cell}
:id: HNXn4_jCRLMY


```
