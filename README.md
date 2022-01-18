# learned\_optimization: Training and evaluating learned optimizers in JAX

learned\_optimization is a research codebase for training learned
optimizers. It implements hand designed and learned optimizers, tasks to meta-train and meta-test them on, and outer-training algorithms such as ES and PES.

To get started see our [documentation](https://learned-optimization.readthedocs.io/en/latest/).

## Quick Start Colab Notebooks

- Introduction notebook: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part1_Introduction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Creating custom tasks: <a href="https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/Part2_CustomTasks.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The fastest way to get started is to copy the Introduction notebook, and experiment using a free accelerator in colab (go to `Runtime` -> `Change runtime type` in colab to select a TPU or GPU backend).

## Local Installation

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
