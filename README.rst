abopt
=====

abopt (ABstract OPTimizer) - optimization of generic numerical models

.. image:: https://travis-ci.org/bccp/abopt.svg?branch=master
    :target: https://travis-ci.org/bccp/abopt

The package contains two components:

- optimize:
  L-BFGS, TrustRegion, and a bunch of simpler optimizer like gradient descent.

- model: 
  vmad (Virtual Machine Automated Differentiation),
  a differentiable state machine for forward modelling,
  moved to https://github.com/rainwoodman/vmad


This is the second iteration of the design.
The current main interface is in `abopt.abopt2`.

I have a feeling we may need to move to abopt3 at some point; the current way
of dealing with meta-parameters (e.g. trust region radius) is mutable and thus
awkward. But maybe there is no way around it.

