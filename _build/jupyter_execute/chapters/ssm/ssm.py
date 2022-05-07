#!/usr/bin/env python
# coding: utf-8

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
    import jsl
except:
    get_ipython().run_line_magic('pip', 'install git+https://github.com/probml/jsl')
    import jsl

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


# ```{math}
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
# \newcommand{\obs}{\vy}
# \newcommand{\inputs}{\vu}
# \newcommand{\input}{\inputs}
# 
# \newcommand{\hmmTrans}{\vA}
# \newcommand{\hmmObs}{\vB}
# \newcommand{\hmmInit}{\vpi}
# \newcommand{\hmmhid}{\hidden}
# \newcommand{\hmmobs}{\obs}
# 
# \newcommand{\ldsDyn}{\vA}
# \newcommand{\ldsObs}{\vC}
# \newcommand{\ldsDynIn}{\vB}
# \newcommand{\ldsObsIn}{\vD}
# \newcommand{\ldsDynNoise}{\vQ}
# \newcommand{\ldsObsNoise}{\vR}
# 
# \newcommand{\ssmDyn}{f}
# \newcommand{\ssmObs}{h}
# ```
# 

# (sec:ssm-intro)=
# # What are State Space Models?
# 
# 
# A state space model or SSM
# is a partially observed Markov model,
# in which the hidden state,  $\hidden_t$,
# evolves over time according to a Markov process,
# possibly conditional on external inputs or controls $\input_t$,
# and each hidden state generates some
# observations $\obs_t$ at each time step.
# (In this book, we mostly focus on discrete time systems,
# although  we consider the continuous-time case in  XXX.)
# We get to see the observations, but not the hidden state.
# Our main goal is to infer the hidden state given the observations.
# However, we can also use the model to predict future observations,
# by first predicting future hidden states, and then predicting
# what observations they might generate.
# By using  a hidden state $\hidden_t$
# to represent the past observations, $\obs_{1:t-1}$,
# the  model can have ``infinite'' memory,
# unlike a standard Markov model.
# 
# Formally we can define an SSM 
# as the following joint distribution:
# ```{math}
# :label: SSMfull
# p(\hmmobs_{1:T},\hmmhid_{1:T}|\inputs_{1:T})
#  = \left[ p(\hmmhid_1|\inputs_1) \prod_{t=2}^{T}
#  p(\hmmhid_t|\hmmhid_{t-1},\inputs_t) \right]
#  \left[ \prod_{t=1}^T p(\hmmobs_t|\hmmhid_t, \inputs_t, \hmmobs_{t-1}) \right]
# ```
# where $p(\hmmhid_t|\hmmhid_{t-1},\inputs_t)$ is the
# transition model,
# $p(\hmmobs_t|\hmmhid_t, \inputs_t, \hmmobs_{t-1})$ is the
# observation model,
# and $\inputs_{t}$ is an optional input or action.
# See {numref}`Figure %s <ssm-ar>` 
# for an illustration of the corresponding graphical model.
# 
# 
# ```{figure} /figures/SSM-AR-inputs.png
# :scale: 100%
# :name: ssm-ar
# 
# Illustration of an SSM as a graphical model.
# ```
# 
# 
# 
# 

# We often consider a simpler setting in which there
# are no external inputs,
# and the observations are conditionally independent of each other
# (rather than having Markovian dependencies) given the hidden state.
# In this case the joint simplifies to 
# ```{math}
# :label: SSMsimplified
# p(\hmmobs_{1:T},\hmmhid_{1:T})
#  = \left[ p(\hmmhid_1) \prod_{t=2}^{T}
#  p(\hmmhid_t|\hmmhid_{t-1}) \right]
#  \left[ \prod_{t=1}^T p(\hmmobs_t|\hmmhid_t \right]
# ```
# See {numref}`Figure %s <ssm-simplified>` 
# for an illustration of the corresponding graphical model.
# Compare {eq}`SSMfull` and {eq}`SSMsimplified`.
# 
# 
# ```{figure} /figures/SSM-simplified.png
# :scale: 100%
# :name: ssm-simplified
# 
# Illustration of a simplified SSM.
# ```

# (sec:hmm-intro)=
# # Hidden Markov Models
# 
# In this section, we discuss the
# hidden Markov model or HMM,
# which is a state space model in which the hidden states
# are discrete, so $\hmmhid_t \in \{1,\ldots, K\}$.
# The observations may be discrete,
# $\hmmobs_t \in \{1,\ldots, C\}$,
# or continuous,
# $\hmmobs_t \in \real^D$,
# or some combination,
# as we illustrate below.
# More details can be found in e.g., 
# {cite}`Rabiner89,Fraser08,Cappe05`.
# For an interactive introduction,
# see https://nipunbatra.github.io/hmm/.

# (sec:casino-ex)=
# ### Example: Casino HMM
# 
# To illustrate HMMs with categorical observation model,
# we consider the "Ocassionally dishonest casino" model from {cite}`Durbin98`.
# There are 2 hidden states, representing whether the dice being used in the casino is fair or loaded.
# Each state defines a distribution over the 6 possible observations.
# 
# The transition model is denoted by
# ```{math}
# p(z_t=j|z_{t-1}=i) = \hmmTrans_{ij}
# ```
# Here the $i$'th row of $\vA$ corresponds to the outgoing distribution from state $i$.
# This is  a row stochastic matrix,
# meaning each row sums to one.
# We can visualize
# the non-zero entries in the transition matrix by creating a state transition diagram,
# as shown in {ref}`casino-fig`.
# 
# ```{figure} /figures/casino.png
# :scale: 50%
# :name: casino-fig
# 
# Illustration of the casino HMM.
# ```
# 
# The  observation model
# $p(\obs_t|\hiddden_t=j)$ has the form
# ```{math}
# p(\obs_t=k|\hidden_t=j) = \hmmObs_{jk} 
# ```
# This is represented by the histograms associated with each
# state in {ref}`casino-fig`.
# 
# Finally,
# the initial state distribution is denoted by
# ```{math}
# p(z_1=j) = \hmmInit_j
# ```
# 
# Collectively we denote all the parameters by $\vtheta=(\hmmTrans, \hmmObs, \hmmInit)$.
# 
# Now let us implement this model code.

# In[3]:


# state transition matrix
A = np.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

# observation matrix
B = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

pi = np.array([0.5, 0.5])

(nstates, nobs) = np.shape(B)

