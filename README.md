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

The propose algorithm approximates a high dimensional function $u : \mathcal{X} \subset \R^d \rightarrow \R^q$ by the composition of a \emph{feature map} $g : \mathcal{X} \rightarrow \R^m$ and a \emph{profile function} $f : \R^m \rightarrow \R^q$ with \emph{latent dimension} $m\leq d$. The goal is to ensure that
$$
    \label{approx_error}
	\E[\|u(\mathbf{X})-f\circ g(\mathbf{X})\|^2] \leq \varepsilon ,
$$
for some prescribed tolerance $\varepsilon$, $\E[\cdot]$ denoting the expectation and where $\|\cdot\|$ denotes the Euclidean norm on $\R^q$.
The feature map $g$ is sought in a set $\mathcal{G}_m$ defined as
$$
	\mathcal{G}_m =
	\left\{ \left. \begin{array}{lcll}
		g : & \mathcal{X} & \rightarrow & \R^m \\
		\quad & x
		& \mapsto &
		(\varphi_1(x), \ldots, \varphi_m(x))\\
	\end{array} \right|
	\varphi
	\in \mathcal{D}
	\right\} ,
$$
where $\mathcal{D}$ is a set of tractable $\mathcal{C}^1$-diffeomorphisms from $\mathcal{X}$ to $\R^d$. In practice $\mathcal{D}$ is a set of invertible neural networks, \emph{e.g.} block affine coupling flows. 
$\mathcal{X}$ is referred as the \emph{input space (IS)} and $\varphi(\mathcal{X})$ is referred as the \emph{feature space}.

The algorithm first learns the feature map $g$ by minimizing an upper-bound of the approximation error \ref{approx_error} and then build the profile functin $f$ by regressing $u(\bm{X})$ againts $g(\bm{X})$. The upper-bound depends on $\nabla g$ and $\nabla u$ which requires access to gradient samples of the model. 

## Invertible neural networks: Block Affine Coupling Flows (BACF)

For $d\geq2$ and for two functions $s, t :\mathbb{R}^{\lfloor d/2 \rfloor}\rightarrow\mathbb{R}^{d-\lfloor d/2 \rfloor} $,
	the block affine coupling flow (BACF) $\Psi_{s, t} : \mathbb{R}^d \rightarrow \mathbb{R}^d$ is defined by

$$
	\Psi_{s,t}(x) =
	\begin{pmatrix}
	x_{\leq \lfloor d/2 \rfloor} \\  x_{>\lfloor d/2 \rfloor}\odot \exp(s(x_{\leq \lfloor d/2 \rfloor}))+ t(x_{\leq \lfloor d/2 \rfloor})
	\end{pmatrix},
$$
where $\odot$ is the element-wise product and where $\exp(\cdot)$ is applied element-wise. In our implementation $s$ and $t$ are fully connected neural networks implemented in src/coupling_functions.py.

By composing multiple BACF blocks and for $d,\ell\in\mathbb{N}^*$, we obtain the following sets of diffeomorphisms:
 \begin{align*}
  \mathcal{D}^{d,\ell}_{\mathrm{BACF}} &= \left\{ (P \circ \Psi_{s_\ell,t_\ell}) \circ \ldots \circ ( P \circ \Psi_{s_1,t_1}) ~\Big|~ s_i,t_i \in \mathcal{C}^1(\mathbb{R}^{\lfloor \frac{d}{2} \rfloor};\mathbb{R}^{d-\lfloor \frac{d}{2} \rfloor})
   \right\}
 \end{align*}
 where $P$ is the block-coordinate permutation defined by $P(x) = (x_{>\lfloor \frac{d}{2} \rfloor}  ,   x_{\leq \lfloor \frac{d}{2} \rfloor})$ for $x\in\mathbb{R}^d$.

In our implementation we set $\mathcal{D} = \mathcal{D}^{d,\ell}_{\mathrm{BACF}}$ for a fixed number of layer $\ell$. Single BACF layers are implemented in the class BACF_block and BACF invertible neural networks are implemented in the class BACF_invnet in src/bacf_invnet.py.
---


