#!/usr/bin/env python
# coding: utf-8

# (sec:forwards)=
# # HMM filtering (forwards algorithm)

# In[1]:


### Import standard libraries

import abc
from dataclasses import dataclass
import functools
from functools import partial
import itertools
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, NamedTuple, Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad
#from jax.scipy.special import logit
#from jax.nn import softmax
import jax.random as jr



import distrax
import optax

import jsl
import ssm_jax


# 
# ## Introduction
# 
# 
# The  $\keyword{Bayes filter}$ is an algorithm for recursively computing
# the belief state
# $p(\hidden_t|\obs_{1:t})$ given
# the prior belief from the previous step,
# $p(\hidden_{t-1}|\obs_{1:t-1})$,
# the new observation $\obs_t$,
# and the model.
# This can be done using $\keyword{sequential Bayesian updating}$.
# For a dynamical model, this reduces to the
# $\keyword{predict-update}$ cycle described below.
# 
# The $\keyword{prediction step}$ is just the $\keyword{Chapman-Kolmogorov equation}$:
# \begin{align}
# p(\hidden_t|\obs_{1:t-1})
# = \int p(\hidden_t|\hidden_{t-1}) p(\hidden_{t-1}|\obs_{1:t-1}) d\hidden_{t-1}
# \end{align}
# The prediction step computes
# the $\keyword{one-step-ahead predictive distribution}$
# for the latent state, which converts
# the posterior from the previous time step to become the prior
# for the current step.
# 
# 
# The $\keyword{update step}$
# is just Bayes rule:
# \begin{align}
# p(\hidden_t|\obs_{1:t}) = \frac{1}{Z_t}
# p(\obs_t|\hidden_t) p(\hidden_t|\obs_{1:t-1})
# \end{align}
# where the normalization constant is
# \begin{align}
# Z_t = \int p(\obs_t|\hidden_t) p(\hidden_t|\obs_{1:t-1}) d\hidden_{t}
# = p(\obs_t|\obs_{1:t-1})
# \end{align}
# 
# Note that we can derive the log marginal likelihood from these normalization constants
# as follows:
# ```{math}
# :label: eqn:logZ
# 
# \log p(\obs_{1:T})
# = \sum_{t=1}^{T} \log p(\obs_t|\obs_{1:t-1})
# = \sum_{t=1}^{T} \log Z_t
# ```
# 
# 

# 
# When the latent states $\hidden_t$ are discrete, as in HMM,
# the above integrals become sums.
# In particular, suppose we define
# the $\keyword{belief state}$ as $\alpha_t(j) \defeq p(\hidden_t=j|\obs_{1:t})$,
# the  $\keyword{local evidence}$ (or $\keyword{local likelihood}$)
# as $\lambda_t(j) \defeq p(\obs_t|\hidden_t=j)$,
# and the transition matrix as
# $\hmmTrans(i,j)  = p(\hidden_t=j|\hidden_{t-1}=i)$.
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
# = \frac{1}{Z_t} \lambda_t(j) \left[\sum_i \alpha_{t-1}(i) \hmmTrans(i,j)  \right]
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
# 

# 
# Since all the quantities are finite length vectors and matrices,
# we can implement the whole procedure using matrix vector multoplication:
# ```{math}
# :label: eqn:fwdsAlgoMatrixForm
# \valpha_t =\text{normalize}\left(
# \vlambda_t \dotstar  (\hmmTrans^{\trans} \valpha_{t-1}) \right)
# ```
# where $\dotstar$ represents
# elementwise vector multiplication,
# and the $\text{normalize}$ function just ensures its argument sums to one.
# 

# ## Example
# 
# In {ref}`sec:casino-inference`
# we illustrate
# filtering for the casino HMM,
# applied to a random sequence $\obs_{1:T}$ of length $T=300$.
# In blue, we plot the probability that the dice is in the loaded (vs fair) state,
# based on the evidence seen so far.
# The gray bars indicate time intervals during which the generative
# process actually switched to the loaded dice.
# We see that the probability generally increases in the right places.

# ## Normalization constants
# 
# In most publications on HMMs,
# such as {cite}`Rabiner89`,
# the forwards message is defined
# as the following unnormalized joint probability:
# ```{math}
# \alpha'_t(j) = p(\hidden_t=j,\obs_{1:t}) 
# = \lambda_t(j) \left[\sum_i \alpha'_{t-1}(i) A(i,j)  \right]
# ```
# In this book we define the forwards message   as the normalized
# conditional probability
# ```{math}
# \alpha_t(j) = p(\hidden_t=j|\obs_{1:t}) 
# = \frac{1}{Z_t} \lambda_t(j) \left[\sum_i \alpha_{t-1}(i) A(i,j)  \right]
# ```
# where $Z_t = p(\obs_t|\obs_{1:t-1})$.
# 
# The "traditional" unnormalized form has several problems.
# First, it rapidly suffers from numerical underflow,
# since the probability of
# the joint event that $(\hidden_t=j,\obs_{1:t})$
# is vanishingly small. 
# To see why, suppose the observations are independent of the states.
# In this case, the unnormalized joint has the form
# \begin{align}
# p(\hidden_t=j,\obs_{1:t}) = p(\hidden_t=j)\prod_{i=1}^t p(\obs_i)
# \end{align}
# which becomes exponentially small with $t$, because we multiply
# many probabilities which are less than one.
# Second, the unnormalized probability is less interpretable,
# since it is a joint distribution over states and observations,
# rather than a conditional probability of states given observations.
# Third, the unnormalized joint form is harder to approximate
# than the normalized form.
# Of course,
# the two definitions only differ by a
# multiplicative constant
# {cite}`Devijver85`,
# so the algorithmic difference is just
# one line of code (namely the presence or absence of a call to the `normalize` function).
# 
# 
# 
# 
# 

# ## Naive implementation
# 
# Below we give a simple numpy implementation of the forwards algorithm.
# We assume the HMM uses categorical observations, for simplicity.
# 
# 

# In[2]:




def normalize_np(u, axis=0, eps=1e-15):
    u = np.where(u == 0, 0, np.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = np.where(c == 0, 1, c)
    return u / c, c

def hmm_forwards_np(trans_mat, obs_mat, init_dist, obs_seq):
    n_states, n_obs = obs_mat.shape
    seq_len = len(obs_seq)

    alpha_hist = np.zeros((seq_len, n_states))
    ll_hist = np.zeros(seq_len)  # loglikelihood history

    alpha_n = init_dist * obs_mat[:, obs_seq[0]]
    alpha_n, cn = normalize_np(alpha_n)

    alpha_hist[0] = alpha_n
    log_normalizer = np.log(cn)

    for t in range(1, seq_len):
        alpha_n = obs_mat[:, obs_seq[t]] * (alpha_n[:, None] * trans_mat).sum(axis=0)
        alpha_n, zn = normalize_np(alpha_n)

        alpha_hist[t] = alpha_n
        log_normalizer = np.log(zn) + log_normalizer

    return  log_normalizer, alpha_hist


# ## Numerically stable implementation 
# 
# 
# 
# In practice it is more numerically stable to compute
# the log likelihoods $\ell_t(j) = \log p(\obs_t|\hidden_t=j)$,
# rather than the likelioods $\lambda_t(j) = p(\obs_t|\hidden_t=j)$.
# In this case, we can perform the posterior updating in a numerically stable way as follows.
# Define $L_t = \max_j \ell_t(j)$ and
# \begin{align}
# \tilde{p}(\hidden_t=j,\obs_t|\obs_{1:t-1})
# &\defeq p(\hidden_t=j|\obs_{1:t-1}) p(\obs_t|\hidden_t=j) e^{-L_t} \\
#  &= p(\hidden_t=j|\obs_{1:t-1}) e^{\ell_t(j) - L_t}
# \end{align}
# Then we have
# \begin{align}
# p(\hidden_t=j|\obs_t,\obs_{1:t-1})
#   &= \frac{1}{\tilde{Z}_t} \tilde{p}(\hidden_t=j,\obs_t|\obs_{1:t-1}) \\
# \tilde{Z}_t &= \sum_j \tilde{p}(\hidden_t=j,\obs_t|\obs_{1:t-1})
# = p(\obs_t|\obs_{1:t-1}) e^{-L_t} \\
# \log Z_t &= \log p(\obs_t|\obs_{1:t-1}) = \log \tilde{Z}_t + L_t
# \end{align}
# 
# Below we show some JAX code that implements this core operation.
# 

# In[3]:



def _condition_on(probs, ll):
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    norm = new_probs.sum()
    new_probs /= norm
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


# With the above function, we can implement a more numerically stable version of the forwards filter,
# that works for any likelihood function, as shown below. It takes in the prior predictive distribution,
# $\alpha_{t|t-1}$,
# stored in `predicted_probs`, and conditions them on the log-likelihood for each time step $\ell_t$ to get the
# posterior, $\alpha_t$, stored in `filtered_probs`,
# which is then converted to the prediction for the next state, $\alpha_{t+1|t}$.

# In[4]:


def _predict(probs, A):
    return A.T @ probs


def hmm_filter(initial_distribution,
               transition_matrix,
               log_likelihoods):
    def _step(carry, t):
        log_normalizer, predicted_probs = carry

        # Get parameters for time t
        get = lambda x: x[t] if x.ndim == 3 else x
        A = get(transition_matrix)
        ll = log_likelihoods[t]

        # Condition on emissions at time t, being careful not to overflow
        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        # Update the log normalizer
        log_normalizer += log_norm
        # Predict the next state
        predicted_probs = _predict(filtered_probs, A)

        return (log_normalizer, predicted_probs), (filtered_probs, predicted_probs)

    num_timesteps = len(log_likelihoods)
    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps))
    return log_normalizer, filtered_probs, predicted_probs


# 
# TODO: check equivalence of these two implementations!
