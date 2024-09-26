# Land Stewardship and Development Behaviors under an Ecological-Impact Weighted Land Value Tax Scheme: A Proof-of-Concept Agent-Based Model

This repository contains the source code for the agent-based model.

# How to run

## Installations

We suggest working with a virtual Python environment. Our environment can be replicated via

```
conda environment create -f land_dev_environment.yml
```

## Run

After creating the environment with the required packages, you can replicate our results by running

```
cd src
./outputs/run_experiments.sh
```

The outputs will be placed into the ```src/outputs``` folder. ```src/outputs/run_experiments.sh``` and ```src/main.py``` can be examined to see how to run different experiments.
