#!/usr/bin/env python
# coding: utf-8

# (sec:hmm-ex)=
# # Hidden Markov Models
# 
# In this section, we introduce Hidden Markov Models (HMMs).

# ## Boilerplate

# In[1]:


# Install necessary libraries

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


# Import standard libraries

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
from rich import inspect as r_inspect
from rich import print as r_print

def print_source(fname):
    r_print(py_inspect.getsource(fname))


# ## Utility code

# In[3]:




def normalize(u, axis=0, eps=1e-15):
    '''
    Normalizes the values within the axis in a way that they sum up to 1.
    Parameters
    ----------
    u : array
    axis : int
    eps : float
        Threshold for the alpha values
    Returns
    -------
    * array
        Normalized version of the given matrix
    * array(seq_len, n_hidden) :
        The values of the normalizer
    '''
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c


# (sec:casino-ex)=
# ## Example: Casino HMM
# 
# We first create the "Ocassionally dishonest casino" model from {cite}`Durbin98`.
# 
# ```{figure} /figures/casino.png
# :scale: 50%
# :name: casino-fig
# 
# Illustration of the casino HMM.
# ```
# 
# There are 2 hidden states, each of which emit 6 possible observations.

# In[4]:


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

pi, _ = normalize(np.array([1, 1]))
pi = np.array(pi)


(nstates, nobs) = np.shape(B)


# Let's make a little data structure to store all the parameters.
# We use NamedTuple rather than dataclass, since we assume these are immutable.
# (Also, standard python dataclass does not work well with JAX, which requires parameters to be
# pytrees, as discussed in https://github.com/google/jax/issues/2371).

# In[5]:


Array = Union[np.array, jnp.array]

class HMM(NamedTuple):
    trans_mat: Array  # A : (n_states, n_states)
    obs_mat: Array  # B : (n_states, n_obs)
    init_dist: Array  # pi : (n_states)

params_np = HMM(A, B, pi)
print(params_np)
print(type(params_np.trans_mat))


params = jax.tree_map(lambda x: jnp.array(x), params_np)
print(params)
print(type(params.trans_mat))


# ## Sampling from the joint
# 
# Let's write code to sample from this model. 
# 

# ### Numpy version
# 
# First we code it in numpy using a for loop.

# In[6]:


def hmm_sample_np(params, seq_len, random_state=0):
    np.random.seed(random_state)
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape
    state_seq = np.zeros(seq_len, dtype=int)
    obs_seq = np.zeros(seq_len, dtype=int)
    for t in range(seq_len):
        if t==0:
            zt = np.random.choice(n_states, p=init_dist)
        else:
            zt = np.random.choice(n_states, p=trans_mat[zt])
        yt = np.random.choice(n_obs, p=obs_mat[zt])
        state_seq[t] = zt
        obs_seq[t] = yt

    return state_seq, obs_seq


# In[7]:


seq_len = 100
state_seq, obs_seq = hmm_sample_np(params_np, seq_len, random_state=1)
print(state_seq)
print(obs_seq)


# ### JAX version
# 
# Now let's write a JAX version using jax.lax.scan (for the inter-dependent states) and vmap (for the observations).
# This is harder to read than the numpy version, but faster.

# In[8]:


#@partial(jit, static_argnums=(1,))
def markov_chain_sample(rng_key, init_dist, trans_mat, seq_len):
    n_states = len(init_dist)

    def draw_state(prev_state, key):
        state = jax.random.choice(key, n_states, p=trans_mat[prev_state])
        return state, state

    rng_key, rng_state = jax.random.split(rng_key, 2)
    keys = jax.random.split(rng_state, seq_len - 1)
    initial_state = jax.random.choice(rng_key, n_states, p=init_dist)
    final_state, states = jax.lax.scan(draw_state, initial_state, keys)
    state_seq = jnp.append(jnp.array([initial_state]), states)

    return state_seq


# In[9]:


#@partial(jit, static_argnums=(1,))
def hmm_sample(rng_key, params, seq_len):

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape
    rng_key, rng_obs = jax.random.split(rng_key, 2)
    state_seq = markov_chain_sample(rng_key, init_dist, trans_mat, seq_len)

    def draw_obs(z, key):
        obs = jax.random.choice(key, n_obs, p=obs_mat[z])
        return obs

    keys = jax.random.split(rng_obs, seq_len)
    obs_seq = jax.vmap(draw_obs, in_axes=(0, 0))(state_seq, keys)
    
    return state_seq, obs_seq


# In[10]:


#@partial(jit, static_argnums=(1,))
def hmm_sample2(rng_key, params, seq_len):

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    def draw_state(prev_state, key):
        state = jax.random.choice(key, n_states, p=trans_mat[prev_state])
        return state, state

    rng_key, rng_state, rng_obs = jax.random.split(rng_key, 3)
    keys = jax.random.split(rng_state, seq_len - 1)
    initial_state = jax.random.choice(rng_key, n_states, p=init_dist)
    final_state, states = jax.lax.scan(draw_state, initial_state, keys)
    state_seq = jnp.append(jnp.array([initial_state]), states)

    def draw_obs(z, key):
        obs = jax.random.choice(key, n_obs, p=obs_mat[z])
        return obs

    keys = jax.random.split(rng_obs, seq_len)
    obs_seq = jax.vmap(draw_obs, in_axes=(0, 0))(state_seq, keys)

    return state_seq, obs_seq


# In[11]:



key = PRNGKey(2)
seq_len = 100

state_seq, obs_seq = hmm_sample(key, params, seq_len)
print(state_seq)
print(obs_seq)


# ### Check correctness by computing empirical pairwise statistics
# 
# We will compute the number of i->j transitions, and check that it is close to the true 
# A[i,j] transition probabilites.

# In[12]:


import collections
def compute_counts(state_seq, nstates):
    wseq = np.array(state_seq)
    word_pairs = [pair for pair in zip(wseq[:-1], wseq[1:])]
    counter_pairs = collections.Counter(word_pairs)
    counts = np.zeros((nstates, nstates))
    for (k,v) in counter_pairs.items():
        counts[k[0], k[1]] = v
    return counts

def normalize_counts(counts):
    ncounts = vmap(lambda v: normalize(v)[0], in_axes=0)(counts)
    return ncounts

init_dist = jnp.array([1.0, 0.0])
trans_mat = jnp.array([[0.7, 0.3], [0.5, 0.5]])
rng_key = jax.random.PRNGKey(0)
seq_len = 500
state_seq = markov_chain_sample(rng_key, init_dist, trans_mat, seq_len)
print(state_seq)

counts = compute_counts(state_seq, nstates=2)
print(counts)

trans_mat_empirical = normalize_counts(counts)
print(trans_mat_empirical)

assert jnp.allclose(trans_mat, trans_mat_empirical, atol=1e-1)


# In[ ]:




