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

# (sec:nlds-intro)=
# # Nonlinear Gaussian SSMs
# 
# In this section, we consider SSMs in which the dynamics and/or observation models are nonlinear,
# but the process noise and observation noise are Gaussian.
# That is, 
# \begin{align}
# \hmmhid_t &= \ssmDynFn(\hmmhid_{t-1}, \inputs_t) +  \vepsilon_t  \\
# \hmmobs_t &= \ssmObsFn(\hmmhid_{t}, \inputs_t) + \veta_t
# \end{align}
# where $\vepsilon_t \sim \gauss(\vzero,\vQ_t)$
# and $\veta_t \sim \gauss(\vzero,\vR_t)$.
# This is a very widely used model class. We give some examples below.

# (sec:pendulum)=
# ## Example: tracking a 1d pendulum
# 
# ```{figure} /figures/pendulum.png
# :scale: 50%
# :name: fig:pendulum
# 
# Illustration of a pendulum swinging.
# $g$ is the force of gravity,
# $w(t)$ is a random external force,
# and $\alpha$ is the angle wrt the vertical.
# Based on {cite}`Sarkka13` fig 3.10.
# ```
# 
# 
# % Sarka p45, p74
# Consider a simple pendulum of unit mass and length swinging from
# a fixed attachment, as in
# {numref}`Figure %s <fig:pendulum>`.
# Such an object is in principle entirely deterministic in its behavior.
# However, in the real world, there are often unknown forces at work
# (e.g., air turbulence, friction).
# We will model these by a continuous time random Gaussian noise process $w(t)$.
# This gives rise to the following differential equation:
# \begin{align}
# \frac{d^2 \alpha}{d t^2}
# = -g \sin(\alpha) + w(t)
# \end{align}
# We can write this as a nonlinear SSM by defining the state to be
# $z_1(t) = \alpha(t)$ and $z_2(t) = d\alpha(t)/dt$.
# Thus
# \begin{align}
# \frac{d \vz}{dt}
# = \begin{pmatrix} z_2 \\ -g \sin(z_1) \end{pmatrix}
# + \begin{pmatrix} 0 \\ 1 \end{pmatrix} w(t)
# \end{align}
# If we discretize this step size $\Delta$,
# we get the following
# formulation {cite}`Sarkka13` p74:
# \begin{align}
# \underbrace{
#   \begin{pmatrix} z_{1,t} \\ z_{2,t} \end{pmatrix}
#   }_{\hmmhid_t}
# =
# \underbrace{
#   \begin{pmatrix} z_{1,t-1} + z_{2,t-1} \Delta  \\
#     z_{2,t-1} -g \sin(z_{1,t-1}) \Delta  \end{pmatrix}
#   }_{\vf(\hmmhid_{t-1})}
# +\vq_{t-1}
# \end{align}
# where $\vq_{t-1} \sim \gauss(\vzero,\vQ)$ with
# \begin{align}
# \vQ = q^c \begin{pmatrix}
#   \frac{\Delta^3}{3} &   \frac{\Delta^2}{2} \\
#   \frac{\Delta^2}{2} & \Delta
#   \end{pmatrix}
#   \end{align}
# where $q^c$ is the spectral density (continuous time variance)
# of the continuous-time noise process.
# 
# 
# If we observe the angular position, we
# get the linear observation model
# \begin{align}
# y_t = \alpha_t + r_t =  h(\hmmhid_t) + r_t
# \end{align}
# where $h(\hmmhid_t) = z_{1,t}$
# and $r_t$ is the observation noise.
# If we only observe  the horizontal position,
# we get the nonlinear observation model
# \begin{align}
# y_t = \sin(\alpha_t) + r_t =  h(\hmmhid_t) + r_t
# \end{align}
# where $h(\hmmhid_t) = \sin(z_{1,t})$.
# 
# 
# 
# 
# 
# 
