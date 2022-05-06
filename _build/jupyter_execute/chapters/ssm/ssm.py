#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from rich import inspect as r_inspect
from rich import print as r_print

def print_source(fname):
    r_print(py_inspect.getsource(fname))


# (sec:ssm-intro)=
# # What are State Space Models?
# 
# 
# A state space model or SSM
# is a partially observed Markov model,
# in which the hidden state,  $z_t$,
# evolves over time according to a Markov process.
# 
# 

# ```{figure} /figures/SSM-AR-inputs.png
# :scale: 100%
# :name: ssm-ar
# 
# Illustration of an SSM as a graphical model.
# ```
# 
# ```{figure} /figures/SSM-simplified.png
# :scale: 100%
# :name: ssm-simplifed
# 
# Illustration of a simplified SSM.
# ```

# (sec:casino-ex)=
# ## Example: Casino HMM
# 
# We first create the "Ocassionally dishonest casino" model from {cite}`Durbin98`.
# 
# 
# 
# There are 2 hidden states, each of which emit 6 possible observations.

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

pi, _ = normalize(np.array([1, 1]))
pi = np.array(pi)


(nstates, nobs) = np.shape(B)


# In[ ]:




