abopt
=====

Numerical optimiztion on abstract types

.. image:: https://travis-ci.org/bccp/abopt.svg?branch=master
    :target: https://travis-ci.org/bccp/abopt

This is a translation of the LBFGS and CG optimizer from their C++ version
in CosmoPP to python.

We want the algorithms to take abstract types because we will apply it on
vectors distributed via MPI. Locally each chunk is stored as a numpy array.

