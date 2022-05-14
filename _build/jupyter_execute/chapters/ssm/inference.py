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

# (sec:inference)=
# # Inferential goals
# 
# ```{figure} /figures/inference-problems-tikz.png
# :scale: 30%
# :name: fig:dbn-inference
# 
# Illustration of the different kinds of inference in an SSM.
#  The main kinds of inference for state-space models.
#     The shaded region is the interval for which we have data.
#     The arrow represents the time step at which we want to perform inference.
#     $t$ is the current time,  $T$ is the sequence length,
# $\ell$ is the lag and $h$ is the prediction horizon.
# ```
# 
# 
# 
# Given the sequence of observations, and a known model,
# one of the main tasks with SSMs
# to perform posterior inference,
# about the hidden states; this is also called
# state estimation.
# At each time step $t$,
# there are multiple forms of posterior we may be interested in computing,
# including the following:
# - the filtering distribution
# $p(\hmmhid_t|\hmmobs_{1:t})$
# - the smoothing distribution
# $p(\hmmhid_t|\hmmobs_{1:T})$ (note that this conditions on future data $T>t$)
# - the fixed-lag smoothing distribution
# $p(\hmmhid_{t-\ell}|\hmmobs_{1:t})$ (note that this
# infers $\ell$ steps in the past given data up to the present).
# 
# We may also want to compute the
# predictive distribution $h$ steps into the future:
# ```{math}
# p(\hmmobs_{t+h}|\hmmobs_{1:t})
# = \sum_{\hmmhid_{t+h}} p(\hmmobs_{t+h}|\hmmhid_{t+h}) p(\hmmhid_{t+h}|\hmmobs_{1:t})
# ```
# where the hidden state predictive distribution is
# \begin{align}
# p(\hmmhid_{t+h}|\hmmobs_{1:t})
# &= \sum_{\hmmhid_{t:t+h-1}}
#  p(\hmmhid_t|\hmmobs_{1:t}) 
#  p(\hmmhid_{t+1}|\hmmhid_{t})
#  p(\hmmhid_{t+2}|\hmmhid_{t+1})
# \cdots
#  p(\hmmhid_{t+h}|\hmmhid_{t+h-1})
# \end{align}
# See 
# {numref}`fig:dbn-inference` for a summary of these distributions.
# 
# In addition  to comuting posterior marginals,
# we may want to compute the most probable hidden sequence,
# i.e., the joint MAP estimate
# ```{math}
# \arg \max_{\hmmhid_{1:T}} p(\hmmhid_{1:T}|\hmmobs_{1:T})
# ```
# or sample sequences from the posterior
# ```{math}
# \hmmhid_{1:T} \sim p(\hmmhid_{1:T}|\hmmobs_{1:T})
# ```
# 
# Algorithms for all these task are discussed in the following chapters,
# since the details depend on the form of the SSM.
# 
# 
# 
# 
# 

# (sec:casino-inference)=
# ## Example: inference in the casino HMM
# 
# 
# We now illustrate filtering, smoothing and MAP decoding applied
# to the casino HMM from {ref}`sec:casino` and [](sec:casino). 
# 

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

import distrax
from distrax import HMM


hmm = HMM(trans_dist=distrax.Categorical(probs=A),
            init_dist=distrax.Categorical(probs=pi),
            obs_dist=distrax.Categorical(probs=B))


seed = 314
n_samples = 300
z_hist, x_hist = hmm.sample(seed=PRNGKey(seed), seq_len=n_samples)


# In[4]:


# Call inference engine

filtered_dist, _, smoothed_dist, loglik = hmm.forward_backward(x_hist)
map_path = hmm.viterbi(x_hist)


# In[5]:


# Find the span of timesteps that the    simulated systems turns to be in state 1
def find_dishonest_intervals(z_hist):
    spans = []
    x_init = 0
    for t, _ in enumerate(z_hist[:-1]):
        if z_hist[t + 1] == 0 and z_hist[t] == 1:
            x_end = t
            spans.append((x_init, x_end))
        elif z_hist[t + 1] == 1 and z_hist[t] == 0:
            x_init = t + 1
    return spans


# In[6]:


# Plot posterior
def plot_inference(inference_values, z_hist, ax, state=1, map_estimate=False):
    n_samples = len(inference_values)
    xspan = np.arange(1, n_samples + 1)
    spans = find_dishonest_intervals(z_hist)
    if map_estimate:
        ax.step(xspan, inference_values, where="post")
    else:
        ax.plot(xspan, inference_values[:, state])

    for span in spans:
        ax.axvspan(*span, alpha=0.5, facecolor="tab:gray", edgecolor="none")
    ax.set_xlim(1, n_samples)
    # ax.set_ylim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Observation number")


# In[7]:


# Filtering
fig, ax = plt.subplots()
plot_inference(filtered_dist, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Filtered")


 


# In[8]:


# Smoothing
fig, ax = plt.subplots()
plot_inference(smoothed_dist, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")


 


# In[9]:


# MAP estimation
fig, ax = plt.subplots()
plot_inference(map_path, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")


# In[10]:


# TODO: posterior samples


# ## Example: inference in the tracking LG-SSM
# 
# We now illustrate filtering, smoothing and MAP decoding applied
# to the 2d tracking HMM from {ref}`sec:tracking-lds`. 

# In[11]:


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
mu0 = jnp.array([8, 10, 1, 0]).astype(float)
Sigma0 = jnp.eye(state_size) * 1.0

from jsl.lds.kalman_filter import LDS, smooth, filter

lds = LDS(A, C, Q, R, mu0, Sigma0)


z_hist, x_hist = lds.sample(key, timesteps)


# In[12]:


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


# In[13]:


# Filtering

mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = filter(lds, x_hist)
l2_filter = jnp.linalg.norm(z_hist[:, :2] - mu_hist[:, :2], 2)
print(f"L2-filter: {l2_filter:0.4f}")
fig_filtered, axs = plt.subplots()
plot_tracking_values(x_hist, mu_hist, Sigma_hist, "filtered", axs)


# In[14]:


# Smoothing

mu_hist_smooth, Sigma_hist_smooth = smooth(lds, mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist)
l2_smooth = jnp.linalg.norm(z_hist[:, :2] - mu_hist_smooth[:, :2], 2)
print(f"L2-smooth: {l2_smooth:0.4f}")
fig_smoothed, axs = plt.subplots()
plot_tracking_values(x_hist, mu_hist_smooth, Sigma_hist_smooth, "smoothed", axs)

