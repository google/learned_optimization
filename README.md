# learned\_optimization: Training and evaluating learned optimizers in JAX

learned\_optimization is a research codebase for training learned
optimizers. It implements hand designed and learned optimizers, tasks to meta-train and meta-test them on, and outer-training algorithms such as ES and PES.

## Quick Start Colab Notebooks

The fastest way to get started is to fork a [colab notebook](https://github.com/google/learned_optimization/notebooks) and leverage a free accelerator there.

## Installation

We strongly recommend using virtualenv to work with this package.

```
pip3 install virtualenv
git clone git@github.com:google/learned_optimizers.git
cd learned_optimizers
python3 -m venv env
source env/bin/activate
pip install -e .
```

Then run the tests to make sure everything is functioning properly.

```
python3 -m nose
```

If something is broken please file an issue and we will take a look!

## Disclaimer

learned\_optimization is not an official Google product.
