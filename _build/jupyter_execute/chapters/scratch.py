#!/usr/bin/env python
# coding: utf-8

# (ch:intro)=
# # Scratchpad
# 
# 
# In this chapter, we do blah.
# Specifically
# 
# - foo
# - bar.
# - baz
# 
# For more details, see 
# {ref}`ch:hmm` and  {cite}`Sarkka13`.
# 
# 
# ## Python
# 
# We\'re now ready to start coding.

# In[1]:


from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


# In[2]:


# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

print(jax.devices())


# ## Images
# 
# 
# <!---
# ![](https://myst-parser.readthedocs.io/en/latest/_static/logo-wide.svg)
# 
# 
# <img src="https://github.com/probml/probml-notebooks/blob/main/images/cat_dog.jpg"
# style="height:200">
# -->
# 
# ```{figure} /figures/cat_dog.jpg
# :scale: 50%
# :name: cat_dog
# 
# A photo of a cat and a dog.
# ```
# 
# ```{figure} /figures/cat_dog.jpg
# :height: 300px
# :name: cat_dog2
# 
# Another photo of a cat and a dog.
# ```
# 
# In {numref}`cat_dog` 
# we show catdog.
# In {numref}`Figure %s <cat_dog2>` we show catdog2, its twin.
# 
# ```{note}
# I am a useful note!
# ```
# 
# ## Math
# 
# Here is $\N=10$ and blah. $\floor{42.3}= 42$. Let's try again.
# 
# We have $E= mc^2$, and also
# 
# ```{math}
# :label: foo
# a x^2 + bx+ c = 0
# ```
# 
# From {eq}`foo`, it follows that
# 
# $$
# \begin{align}
# 0 &= a x^2 + bx+ c \\
# 0 &= a x^2 + bx+ c 
# \end{align}
# $$
