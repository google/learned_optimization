Population: Online hparam adjustment for multiple workers
==========================================================

A micro library for population based training and online hparam adjustment.

Often times, one seeks to modify hyperparameters *while* training some underlying model.
One of the most common type of modification used in a variety of settings are fixed schedules.
These include modifying the learning rate (e.g. learning rate warm up or decay),
to more complex schedules involving curriculum over say data augmentation.
Fixed schedules are difficult to tune, however, and as such this library also supports interactive
hyperparameter updating based on a generalization of population based training.

Quick Start Example
-------------------

This example shows how to leverage population based training training with 2 populations. For a full example see below.

.. code-block::

  from learned_optimization.population import population as pop_mod
  from learned_optimization.population.mutators import winner_take_all_genetic

  # First we define how we want the population to be mutated.
  # Let's just randomly perturb it.
  def mutate_fn(hparams):
    return {"log_lr": hparams["log_lr"] + np.random.normal()*0.1}

  # Select a mutator. This mutator does a population wide selection every 100 steps.
  mutator = winner_take_all_genetic.WinnerTakeAllGenetic(mutate_fn, steps_per_exploit=100)

  # Construct the PopulationController which manages the population
  population = pop_mod.PopulationController([{"log_lr": np.log(1e-4)} for i in range(2)], mutator)

  # Now, on each worker, we train a model.
  gen_id = ""  # The generation we are currently training. No generation at init.
  hparams = None # The initial hparams -- these will be determined by the population
  params = None # No initial parameter values.
  step = 0

  while True:
    # Get the current set of parameters and hparams to work on.
    # Worker id here contains the integer worker index to differentiate the 2 workers.
    new_data = population.maybe_get_worker_data(worker_id, gen_id, step, params, meta_params)
    # If this is non-None then we obtained some new hparams and params to work with.
    if new_data:
      params = new_data.params
      hparams = new_data.meta_params
      gen_id = new_data.generation_id
      step = new_data.step
      if params is None:
        params = init_params()

    # Train the model one step
    params = train_model_one_step(params, hparams)
    step += 1 # accumulate step counter.

    # every 10 steps, send back an evaluation.
    if step % 10 == 0:
      loss = loss_fn(params)
      population.set_eval(worker_id, gen_id, step, params, loss)

How it works
-------------

At it's core, these are all meta-learning based methods for updating hyper parameters.

As such, there are 2 subsets of parameters:
  * hyper parameters (or meta-parameters): Which change slowly. The changes of these values are governed by this library.
  * parameters (or inner-parameters): which change much faster and are often updated many times for each hparam update. In the context of this library, these are often checkpoint paths. 
    This library does not touch updating these parameters.
    
For every new set of hyper parameters, we define a new generation_id. This id is a string, and is unique to this incarnation of the hparam.
Within a single generation_id, we continue to train the parameters and periodically send back evaluations, or some performance measurement obtained with these parameters.
Behind the scenes, each one of these evaluations is stored in a `Checkpoint` object which keeps track of the parameter-iteration (or training step) with which the evaluation / model was done. This forms a graph of the sequence of hyperparameters used to train models.

These checkpoints are stored on the `PopulationController` which provides an interface to build histories of trainings and managing book keeping in a fault tolerant way. This history is stored in a nested dictionary first keyed by a generation_id, then by step stores `Checkpoint` objects.
The decision to branch at a checkpoint is governed by a `Mutate` class which has access to the currently running workers, and the checkpoint cache.


Available Mutators
------------------

Winner take all genetic algorithms
***********************************

This trains each worker of the population for some amount of time (either measured in steps, or wallclock). Once this time has elapsed, and an evaluation is has been sent back for each worker,
all workers are compared. The best performing worker is selected and the parameters and hyper-parameters of this worker are copied to the other workers, and the hyper-parameters are modified.

This is a simplified setup to that explored in the `population based training paper <https://arxiv.org/abs/1711.09846>`_.

Fixed schedule
***********************************
This trains a single worker and modifies hyper parameters in a fixed schedule specified at construction.

Single worker Explore
***********************************
This trains a single worker in alternating explore phases which tries hyperparameters, and exploit phases which uses the best performing hyper parameters.


Examples
---------
We provide 3 more complete examples.

Synthetic
*********
A synthetic problem to quickly demonstrate API

Simple\_CNN
***********
A small CNN trained in multiple threads on cifar10. The parameters are neural network weights (saved to disk) and the hyper parameters adjusted online consist of a single learning rate.

Complex\_CNN
************
A more complicated example of a CNN meta-training many more hparams modifying data aug with a performance measurement targeting validation loss.
