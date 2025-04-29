# Certifiable-tools

A repo for collecting tools common to many certifiable optimization algorithms. 

This includes SDP solvers, low-rank projections, etc. 

## Installation

To install, run
```
conda env create -f environment.yml
```
and check installation by running
```
pytest .
```

## How to use

### Create problems for sparsity-exploiting solvers

See `_tests/test_homqcqp_ineq` for an example for how to easily build and solve a 
problem exploiting sparsity. 
