# Diffeomorphism based feature learning

Official implementation of

> Diffeomorphism-based feature learning using Poincaré inequalities on augmented input space
> Romain Verdière, Clémentine Prieur, Olivier Zahm
> Journal of Machine Learning Research (JMLR), 2026

[[Paper]](https://www.jmlr.org/papers/volume26/23-1707/23-1707.pdf)

---

## Overview

This repository contains the official implementation of the methods presented in our JMLR paper.

The project provides

- dataset generation code
- training and evaluation code
- visualization utilities

Our method learns a nonlinear feature map $g : \mathcal{X} \rightarrow \mathbb{R}^m$ 
and a profile function $f : \mathbb{R}^m \rightarrow \mathbb{R}^q$ 
such that a target function $u : \mathcal{X} \rightarrow \mathbb{R}^q$ 
can be accurately approximated by the composition

$u(x) \approx f(g(x))$

for the $L^2$ norm. Here the latent dimension satisfies $m \ll d$.

The feature map is constrained to be the first coordinates of a learned
diffeomorphism implemented as an invertible neural network based on
Block Affine Coupling Flows (BACFs).

For the complete mathematical formulation, see Section 2 and 3 of the paper.

## Installation

The thermal-block dataset generator relies on
[DOLFINx](https://github.com/FEniCS/dolfinx),MPI and PETSc. We recommend
installing the dependencies with Conda through the `conda-forge` channel.

Clone the repository:

```bash
git clone https://github.com/rverdiere/diffeo_feature_learning.git
cd diffeo_feature_learning
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate diffeo_feature_learning
```

Verify the installation:

```bash
python -c "import dolfinx, mpi4py, numpy, matplotlib, skopt, torch; print('Installatcon successful')"
```
## How to use

The dataset generators can be run with: 
```bash 
python datasets/generate_dataset.py 
python datasets/generate_dataset_thermalblock.py 
python datasets/generate_dataset_ae.py 
```

The experiments from Sections 7 can be run with: 
```bash 
python main.py
python main_autoencoders.py
```

## Repository Structure

```text
diffeo_feature_learning/
├── datasets/           
│   ├── autoencoders/   
│   │   ├── q8.csv      # Q8 values as defined in Appendix F
│   │   └── q16.csv     # Q16 values as defined in Appendix F
│   ├── benchmark_function.py               # Definition of benchmark function from Section 7.1
│   ├── generate_dataset.py                 # Dataset generation script for scalar valued benchmark function
│   ├── generate_dataset_thermalblock.py    # Dataset generation script for the thermalblock model
│   └── generate_dataset_ae.py              # Dataset generation script for the autoencoder example
├── results/           # Results of experiments in csv format
├── src/
│   ├── bacf_invnet.py          # Block Affine Coupling Flow implementation
│   ├── coupling_functions.py   # Coupling layer neural networks
│   ├── losses.py               # Loss functions
│   ├── training_functions.py   # Training functions
│   ├── plot.py                 # Visualization utilities
│   └── preprocess_functions.py # Preprocessing functions (normalization, dataset loading ...)
├── main.py                 # Main file to run the function approximation experiments
├── main_autoencoders.py    # Main file to run the autoencoder experiments
├── environment.yml         # Conda environment 
└── README.md
```

### Main Components

- **`datasets/`** contains utilities for generating the synthetic datasets used in the paper.
- **`src/`** contains the core implementation of the proposed algorithms including invertible neural networks and Poincaré loss functions.
- **`results/`** contains the simulation results in csv format.
- **`main.py`** Script to run the expiriments from Sections 7.1 and 7.2
- **`main_autoencoderes.py`** Script to run the expiriments from Sections 7.3

