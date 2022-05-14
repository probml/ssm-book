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
    import distrax
except:
    get_ipython().run_line_magic('pip', 'install --upgrade  distrax')
    import distrax

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
# \newcommand{\hid}{\hidden}
# \newcommand{\observed}{\vy}
# \newcommand{\obs}{\observed}
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

# (sec:lds-intro)=
# # Linear Gaussian SSMs
# 
# 
# Consider the state space model in 
# {eq}`eq:SSM-ar`
# where we assume the observations are conditionally iid given the
# hidden states and inputs (i.e. there are no auto-regressive dependencies
# between the observables).
# We can rewrite this model as 
# a stochastic nonlinear dynamical system (NLDS)
# by defining the distribution of the next hidden state 
# as a deterministic function of the past state
# plus random process noise $\vepsilon_t$ 
# \begin{align}
# \hmmhid_t &= \ssmDynFn(\hmmhid_{t-1}, \inputs_t, \vepsilon_t)  
# \end{align}
# where $\vepsilon_t$ is drawn from the distribution such
# that the induced distribution
# on $\hmmhid_t$ matches $p(\hmmhid_t|\hmmhid_{t-1}, \inputs_t)$.
# Similarly we can rewrite the observation distributions
# as a deterministic function of the hidden state
# plus observation noise $\veta_t$:
# \begin{align}
# \hmmobs_t &= \ssmObsFn(\hmmhid_{t}, \inputs_t, \veta_t)
# \end{align}
# 
# 
# If we assume additive Gaussian noise,
# the model becomes
# \begin{align}
# \hmmhid_t &= \ssmDynFn(\hmmhid_{t-1}, \inputs_t) +  \vepsilon_t  \\
# \hmmobs_t &= \ssmObsFn(\hmmhid_{t}, \inputs_t) + \veta_t
# \end{align}
# where $\vepsilon_t \sim \gauss(\vzero,\vQ_t)$
# and $\veta_t \sim \gauss(\vzero,\vR_t)$.
# We will call these Gaussian SSMs.
# 
# If we additionally assume
# the transition function $\ssmDynFn$
# and the observation function $\ssmObsFn$ are both linear,
# then we can rewrite the model as follows:
# \begin{align}
# p(\hmmhid_t|\hmmhid_{t-1},\inputs_t) &= \gauss(\hmmhid_t|\ldsDyn_t \hmmhid_{t-1}
# + \ldsDynIn_t \inputs_t, \vQ_t)
# \\
# p(\hmmobs_t|\hmmhid_t,\inputs_t) &= \gauss(\hmmobs_t|\ldsObs_t \hmmhid_{t}
# + \ldsObsIn_t \inputs_t, \vR_t)
# \end{align}
# This is called a 
# linear-Gaussian state space model
# (LG-SSM),
# or a
# linear dynamical system (LDS).
# We usually assume the parameters are independent of time, in which case
# the model is said to be time-invariant or homogeneous.
# 

# (sec:tracking-lds)=
# (sec:kalman-tracking)=
# ## Example: tracking a 2d point
# 
# 
# 
# % Sarkkar p43
# Consider an object moving in $\real^2$.
# Let the state be
# the position and velocity of the object,
# $\vz_t =\begin{pmatrix} u_t & \dot{u}_t & v_t & \dot{v}_t \end{pmatrix}$.
# (We use $u$ and $v$ for the two coordinates,
# to avoid confusion with the state and observation variables.)
# If we use Euler discretization,
# the dynamics become
# \begin{align}
# \underbrace{\begin{pmatrix} u_t\\ \dot{u}_t \\ v_t \\ \dot{v}_t \end{pmatrix}}_{\vz_t}
#   = 
# \underbrace{
# \begin{pmatrix}
# 1 & 0 & \Delta & 0 \\
# 0 & 1 & 0 & \Delta\\
# 0 & 0 & 1 & 0 \\
# 0 & 0 & 0 & 1
# \end{pmatrix}
# }_{\ldsDyn}
# \
# \underbrace{\begin{pmatrix} u_{t-1} \\ \dot{u}_{t-1} \\ v_{t-1} \\ \dot{v}_{t-1} \end{pmatrix}}_{\vz_{t-1}}
# + \vepsilon_t
# \end{align}
# where $\vepsilon_t \sim \gauss(\vzero,\vQ)$ is
# the process noise.
# 
# Let us assume
# that the process noise is 
# a white noise process added to the velocity components
# of the state, but not to the location.
# (This is known as a random accelerations model.)
# We can approximate the resulting process in discrete time by assuming
# $\vQ = \diag(0, q, 0, q)$.
# (See  {cite}`Sarkka13` p60 for a more accurate way
# to convert the continuous time process to discrete time.)
# 
# 
# Now suppose that at each discrete time point we
# observe the location,
# corrupted by  Gaussian noise.
# Thus the observation model becomes
# \begin{align}
# \underbrace{\begin{pmatrix}  y_{1,t} \\  y_{2,t} \end{pmatrix}}_{\vy_t}
#   &=
#     \underbrace{
#     \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 0 & 1 & 0
#     \end{pmatrix}
#     }_{\ldsObs}
#     \
# \underbrace{\begin{pmatrix} u_t\\ \dot{u}_t \\ v_t \\ \dot{v}_t \end{pmatrix}}_{\vz_t}    
#  + \veta_t
# \end{align}
# where $\veta_t \sim \gauss(\vzero,\vR)$ is the \keywordDef{observation noise}.
# We see that the observation matrix $\ldsObs$ simply ``extracts'' the
# relevant parts  of the state vector.
# 
# Suppose we sample a trajectory and corresponding set
# of noisy observations from this model,
# $(\vz_{1:T}, \vy_{1:T}) \sim p(\vz,\vy|\vtheta)$.
# (We use diagonal observation noise,
# $\vR = \diag(\sigma_1^2, \sigma_2^2)$.)
# The results are shown below. 
# 

# In[3]:


key = jax.random.PRNGKey(314)
timesteps = 15
delta = 1.0
A = jnp.array([
    [1, 0, delta, 0],
    [0, 1, 0, delta],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

C = jnp.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

state_size, _ = A.shape
observation_size, _ = C.shape

Q = jnp.eye(state_size) * 0.001
R = jnp.eye(observation_size) * 1.0
# Prior parameter distribution
mu0 = jnp.array([8, 10, 1, 0]).astype(float)
Sigma0 = jnp.eye(state_size) * 1.0

from jsl.lds.kalman_filter import LDS, smooth, filter

lds = LDS(A, C, Q, R, mu0, Sigma0)
print(lds)


# In[4]:


from jsl.demos.plot_utils import plot_ellipse

def plot_tracking_values(observed, filtered, cov_hist, signal_label, ax):
    timesteps, _ = observed.shape
    ax.plot(observed[:, 0], observed[:, 1], marker="o", linewidth=0,
            markerfacecolor="none", markeredgewidth=2, markersize=8, label="observed", c="tab:green")
    ax.plot(*filtered[:, :2].T, label=signal_label, c="tab:red", marker="x", linewidth=2)
    for t in range(0, timesteps, 1):
        covn = cov_hist[t][:2, :2]
        plot_ellipse(covn, filtered[t, :2], ax, n_std=2.0, plot_center=False)
    ax.axis("equal")
    ax.legend()


# In[5]:



z_hist, x_hist = lds.sample(key, timesteps)

fig_truth, axs = plt.subplots()
axs.plot(x_hist[:, 0], x_hist[:, 1],
        marker="o", linewidth=0, markerfacecolor="none",
        markeredgewidth=2, markersize=8,
        label="observed", c="tab:green")

axs.plot(z_hist[:, 0], z_hist[:, 1],
        linewidth=2, label="truth",
        marker="s", markersize=8)
axs.legend()
axs.axis("equal")


# The main task is to infer the hidden states given the noisy
# observations, i.e., $p(\vz|\vy,\vtheta)$. We discuss the topic of inference in {ref}`sec:inference`.
