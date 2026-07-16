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

## Repository Structure

DiffeoFeatureLearning/
├── datasets/          # Synthetic datasets and data generation scripts
├── examples/          # Example notebooks and usage examples
├── results/           # Results of experiments in csv format
├── src/
│   ├── bacf_invnet.py          # Block Affine Coupling Flow implementation
│   ├── coupling_functions.py   # Coupling layer neural networks
│   ├── losses.py               # Loss functions
│   ├── training_functions.py   # Training functions
│   ├── plot.py                 # Visualization utilities
│   └── preprocess_functions.py # Preprocessing functions (normalization, dataset loading ...)
├── main.py            # Main file to run the experiments
├── requirements.txt   # Python dependencies
└── README.md

### Main Components

- **`datasets/`** contains utilities for generating the synthetic datasets used in the paper.
- **`src/`** contains the core implementation of the proposed algorithms including invertible neural networks and Poincaré loss functions.
- **`examples/`** provides simple examples illustrating how to train and evaluate the models.
- **`results/`** contains the simulation results in csv format.

