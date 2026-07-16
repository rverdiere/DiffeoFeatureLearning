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

Our method learns a nonlinear feature map g : X → R^m such that a target function u : X → R^q can be accurately approximated by the composition

u(x) ≈ f(g(x))

where the latent dimension satisfies m ≤ d.

The feature map is constrained to be the first coordinates of a learned
diffeomorphism implemented as an invertible neural network based on
Block Affine Coupling Flows (BACFs).

For the complete mathematical formulation, see Section 2 and 3of the paper.
