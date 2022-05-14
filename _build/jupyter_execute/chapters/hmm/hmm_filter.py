#!/usr/bin/env python
# coding: utf-8

# ```{math}
# 
# \newcommand{\defeq}{\triangleq}
# \newcommand{\trans}{{\mkern-1.5mu\mathsf{T}}}
# \newcommand{\transpose}[1]{{#1}^{\trans}}
# 
# \newcommand{\inv}[1]{{#1}^{-1}}
# \DeclareMathOperator{\dotstar}{\odot}
# 
# 
# \newcommand\floor[1]{\lfloor#1\rfloor}
# 
# \newcommand{\real}{\mathbb{R}}
# 
# % Numbers
# \newcommand{\vzero}{\boldsymbol{0}}
# \newcommand{\vone}{\boldsymbol{1}}
# 
# % Greek https://www.latex-tutorial.com/symbols/greek-alphabet/
# \newcommand{\valpha}{\boldsymbol{\alpha}}
# \newcommand{\vbeta}{\boldsymbol{\beta}}
# \newcommand{\vchi}{\boldsymbol{\chi}}
# \newcommand{\vdelta}{\boldsymbol{\delta}}
# \newcommand{\vDelta}{\boldsymbol{\Delta}}
# \newcommand{\vepsilon}{\boldsymbol{\epsilon}}
# \newcommand{\vzeta}{\boldsymbol{\zeta}}
# \newcommand{\vXi}{\boldsymbol{\Xi}}
# \newcommand{\vell}{\boldsymbol{\ell}}
# \newcommand{\veta}{\boldsymbol{\eta}}
# %\newcommand{\vEta}{\boldsymbol{\Eta}}
# \newcommand{\vgamma}{\boldsymbol{\gamma}}
# \newcommand{\vGamma}{\boldsymbol{\Gamma}}
# \newcommand{\vmu}{\boldsymbol{\mu}}
# \newcommand{\vmut}{\boldsymbol{\tilde{\mu}}}
# \newcommand{\vnu}{\boldsymbol{\nu}}
# \newcommand{\vkappa}{\boldsymbol{\kappa}}
# \newcommand{\vlambda}{\boldsymbol{\lambda}}
# \newcommand{\vLambda}{\boldsymbol{\Lambda}}
# \newcommand{\vLambdaBar}{\overline{\vLambda}}
# %\newcommand{\vnu}{\boldsymbol{\nu}}
# \newcommand{\vomega}{\boldsymbol{\omega}}
# \newcommand{\vOmega}{\boldsymbol{\Omega}}
# \newcommand{\vphi}{\boldsymbol{\phi}}
# \newcommand{\vvarphi}{\boldsymbol{\varphi}}
# \newcommand{\vPhi}{\boldsymbol{\Phi}}
# \newcommand{\vpi}{\boldsymbol{\pi}}
# \newcommand{\vPi}{\boldsymbol{\Pi}}
# \newcommand{\vpsi}{\boldsymbol{\psi}}
# \newcommand{\vPsi}{\boldsymbol{\Psi}}
# \newcommand{\vrho}{\boldsymbol{\rho}}
# \newcommand{\vtheta}{\boldsymbol{\theta}}
# \newcommand{\vthetat}{\boldsymbol{\tilde{\theta}}}
# \newcommand{\vTheta}{\boldsymbol{\Theta}}
# \newcommand{\vsigma}{\boldsymbol{\sigma}}
# \newcommand{\vSigma}{\boldsymbol{\Sigma}}
# \newcommand{\vSigmat}{\boldsymbol{\tilde{\Sigma}}}
# \newcommand{\vsigmoid}{\vsigma}
# \newcommand{\vtau}{\boldsymbol{\tau}}
# \newcommand{\vxi}{\boldsymbol{\xi}}
# 
# 
# % Lower Roman (Vectors)
# \newcommand{\va}{\mathbf{a}}
# \newcommand{\vb}{\mathbf{b}}
# \newcommand{\vBt}{\mathbf{\tilde{B}}}
# \newcommand{\vc}{\mathbf{c}}
# \newcommand{\vct}{\mathbf{\tilde{c}}}
# \newcommand{\vd}{\mathbf{d}}
# \newcommand{\ve}{\mathbf{e}}
# \newcommand{\vf}{\mathbf{f}}
# \newcommand{\vg}{\mathbf{g}}
# \newcommand{\vh}{\mathbf{h}}
# %\newcommand{\myvh}{\mathbf{h}}
# \newcommand{\vi}{\mathbf{i}}
# \newcommand{\vj}{\mathbf{j}}
# \newcommand{\vk}{\mathbf{k}}
# \newcommand{\vl}{\mathbf{l}}
# \newcommand{\vm}{\mathbf{m}}
# \newcommand{\vn}{\mathbf{n}}
# \newcommand{\vo}{\mathbf{o}}
# \newcommand{\vp}{\mathbf{p}}
# \newcommand{\vq}{\mathbf{q}}
# \newcommand{\vr}{\mathbf{r}}
# \newcommand{\vs}{\mathbf{s}}
# \newcommand{\vt}{\mathbf{t}}
# \newcommand{\vu}{\mathbf{u}}
# \newcommand{\vv}{\mathbf{v}}
# \newcommand{\vw}{\mathbf{w}}
# \newcommand{\vws}{\vw_s}
# \newcommand{\vwt}{\mathbf{\tilde{w}}}
# \newcommand{\vWt}{\mathbf{\tilde{W}}}
# \newcommand{\vwh}{\hat{\vw}}
# \newcommand{\vx}{\mathbf{x}}
# %\newcommand{\vx}{\mathbf{x}}
# \newcommand{\vxt}{\mathbf{\tilde{x}}}
# \newcommand{\vy}{\mathbf{y}}
# \newcommand{\vyt}{\mathbf{\tilde{y}}}
# \newcommand{\vz}{\mathbf{z}}
# %\newcommand{\vzt}{\mathbf{\tilde{z}}}
# 
# 
# % Upper Roman (Matrices)
# \newcommand{\vA}{\mathbf{A}}
# \newcommand{\vB}{\mathbf{B}}
# \newcommand{\vC}{\mathbf{C}}
# \newcommand{\vD}{\mathbf{D}}
# \newcommand{\vE}{\mathbf{E}}
# \newcommand{\vF}{\mathbf{F}}
# \newcommand{\vG}{\mathbf{G}}
# \newcommand{\vH}{\mathbf{H}}
# \newcommand{\vI}{\mathbf{I}}
# \newcommand{\vJ}{\mathbf{J}}
# \newcommand{\vK}{\mathbf{K}}
# \newcommand{\vL}{\mathbf{L}}
# \newcommand{\vM}{\mathbf{M}}
# \newcommand{\vMt}{\mathbf{\tilde{M}}}
# \newcommand{\vN}{\mathbf{N}}
# \newcommand{\vO}{\mathbf{O}}
# \newcommand{\vP}{\mathbf{P}}
# \newcommand{\vQ}{\mathbf{Q}}
# \newcommand{\vR}{\mathbf{R}}
# \newcommand{\vS}{\mathbf{S}}
# \newcommand{\vT}{\mathbf{T}}
# \newcommand{\vU}{\mathbf{U}}
# \newcommand{\vV}{\mathbf{V}}
# \newcommand{\vW}{\mathbf{W}}
# \newcommand{\vX}{\mathbf{X}}
# %\newcommand{\vXs}{\vX_{\vs}}
# \newcommand{\vXs}{\vX_{s}}
# \newcommand{\vXt}{\mathbf{\tilde{X}}}
# \newcommand{\vY}{\mathbf{Y}}
# \newcommand{\vZ}{\mathbf{Z}}
# \newcommand{\vZt}{\mathbf{\tilde{Z}}}
# \newcommand{\vzt}{\mathbf{\tilde{z}}}
# 
# 
# %%%%
# \newcommand{\hidden}{\vz}
# \newcommand{\hid}{\hidden}
# \newcommand{\observed}{\vy}
# \newcommand{\obs}{\observed}
# \newcommand{\inputs}{\vu}
# \newcommand{\input}{\inputs}
# 
# \newcommand{\hmmTrans}{\vA}
# \newcommand{\hmmObs}{\vB}
# \newcommand{\hmmInit}{\vpi}
# 
# 
# \newcommand{\ldsDyn}{\vA}
# \newcommand{\ldsObs}{\vC}
# \newcommand{\ldsDynIn}{\vB}
# \newcommand{\ldsObsIn}{\vD}
# \newcommand{\ldsDynNoise}{\vQ}
# \newcommand{\ldsObsNoise}{\vR}
# 
# \newcommand{\ssmDynFn}{f}
# \newcommand{\ssmObsFn}{h}
# 
# 
# %%%
# \newcommand{\gauss}{\mathcal{N}}
# 
# \newcommand{\diag}{\mathrm{diag}}
# ```
# 

# In[1]:


# meta-data does not work yet in VScode
# https://github.com/microsoft/vscode-jupyter/issues/1121

{
    "tags": [
        "hide-cell"
    ]
}


### Install necessary libraries

try:
    import jax
except:
    # For cuda version, see https://github.com/google/jax#installation
    get_ipython().run_line_magic('pip', 'install --upgrade "jax[cpu]"')
    import jax

try:
    import distrax
except:
    get_ipython().run_line_magic('pip', 'install --upgrade  distrax')
    import distrax

try:
    import jsl
except:
    get_ipython().run_line_magic('pip', 'install git+https://github.com/probml/jsl')
    import jsl

#try:
#    import ssm_jax
##except:
#    %pip install git+https://github.com/probml/ssm-jax
#    import ssm_jax

try:
    import rich
except:
    get_ipython().run_line_magic('pip', 'install rich')
    import rich



# In[2]:


{
    "tags": [
        "hide-cell"
    ]
}


### Import standard libraries

import abc
from dataclasses import dataclass
import functools
import itertools

from typing import Any, Callable, NamedTuple, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np


import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad
from jax.scipy.special import logit
from jax.nn import softmax
from functools import partial
from jax.random import PRNGKey, split

import inspect
import inspect as py_inspect
import rich
from rich import inspect as r_inspect
from rich import print as r_print

def print_source(fname):
    r_print(py_inspect.getsource(fname))


# (sec:forwards)=
# # HMM filtering (forwards algorithm)
# 
# 
# The  **Bayes filter** is an algorithm for recursively computing
# the belief state
# $p(\hidden_t|\obs_{1:t})$ given
# the prior belief from the previous step,
# $p(\hidden_{t-1}|\obs_{1:t-1})$,
# the new observation $\obs_t$,
# and the model.
# This can be done using **sequential Bayesian updating**.
# For a dynamical model, this reduces to the
# **predict-update** cycle described below.
# 
# 
# The **prediction step** is just the **Chapman-Kolmogorov equation**:
# ```{math}
# p(\hidden_t|\obs_{1:t-1})
# = \int p(\hidden_t|\hidden_{t-1}) p(\hidden_{t-1}|\obs_{1:t-1}) d\hidden_{t-1}
# ```
# The prediction step computes
# the one-step-ahead predictive distribution
# for the latent state, which updates
# the posterior from the previous time step into the prior
# for the current step.
# 
# 
# The **update step**
# is just Bayes rule:
# ```{math}
# p(\hidden_t|\obs_{1:t}) = \frac{1}{Z_t}
# p(\obs_t|\hidden_t) p(\hidden_t|\obs_{1:t-1})
# ```
# where the normalization constant is
# ```{math}
# Z_t = \int p(\obs_t|\hidden_t) p(\hidden_t|\obs_{1:t-1}) d\hidden_{t}
# = p(\obs_t|\obs_{1:t-1})
# ```
# 
# 
# 
# 

# 
# When the latent states $\hidden_t$ are discrete, as in HMM,
# the above integrals become sums.
# In particular, suppose we define
# the belief state as $\alpha_t(j) \defeq p(\hidden_t=j|\obs_{1:t})$,
# the local evidence as $\lambda_t(j) \defeq p(\obs_t|\hidden_t=j)$,
# and the transition matrix
# $A(i,j)  = p(\hidden_t=j|\hidden_{t-1}=i)$.
# Then the predict step becomes
# ```{math}
# :label: eqn:predictiveHMM
# \alpha_{t|t-1}(j) \defeq p(\hidden_t=j|\obs_{1:t-1})
#  = \sum_i \alpha_{t-1}(i) A(i,j)
# ```
# and the update step becomes
# ```{math}
# :label: eqn:fwdsEqn
# \alpha_t(j)
# = \frac{1}{Z_t} \lambda_t(j) \alpha_{t|t-1}(j)
# = \frac{1}{Z_t} \lambda_t(j) \left[\sum_i \alpha_{t-1}(i) A(i,j)  \right]
# ```
# where
# the  normalization constant for each time step is given by
# ```{math}
# :label: eqn:HMMZ
# \begin{align}
# Z_t \defeq p(\obs_t|\obs_{1:t-1})
# &=  \sum_{j=1}^K p(\obs_t|\hidden_t=j)  p(\hidden_t=j|\obs_{1:t-1}) \\
# &=  \sum_{j=1}^K \lambda_t(j) \alpha_{t|t-1}(j)
# \end{align}
# ```
# 
# Since all the quantities are finite length vectors and matrices,
# we can write the update equation
# in matrix-vector notation as follows:
# ```{math}
# \valpha_t =\text{normalize}\left(
# \vlambda_t \dotstar  (\vA^{\trans} \valpha_{t-1}) \right)
# \label{eqn:fwdsAlgoMatrixForm}
# ```
# where $\dotstar$ represents
# elementwise vector multiplication,
# and the $\text{normalize}$ function just ensures its argument sums to one.
# 
# In {ref}(sec:casino-inference)
# we illustrate
# filtering for the casino HMM,
# applied to a random sequence $\obs_{1:T}$ of length $T=300$.
# In blue, we plot the probability that the dice is in the loaded (vs fair) state,
# based on the evidence seen so far.
# The gray bars indicate time intervals during which the generative
# process actually switched to the loaded dice.
# We see that the probability generally increases in the right places.
# 

# Here is a JAX implementation of the forwards algorithm.

# In[3]:


import jsl.hmm.hmm_lib as hmm_lib
print_source(hmm_lib.hmm_forwards_jax)
#https://github.com/probml/JSL/blob/main/jsl/hmm/hmm_lib.py#L189

