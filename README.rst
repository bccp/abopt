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

The main difference between abopt and scipy's algorithm is that the inner product
and linear operators are supplied via a ``vectorspace`` object. The reason for
this is because on a distributed problem the inner product must do a global
reduction.

The usage involves defining a ``Problem``, then use an optimizer to minimize it.
The test suite are a good source of examples.

Things have been put together in more or less of a haste.
I have a feeling we may need a restructure at some point; the current way
of dealing with meta-parameters (e.g. trust region radius) is mutable and thus
awkward.
