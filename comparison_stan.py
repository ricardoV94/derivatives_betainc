# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# # %load_ext autoreload
# # %autoreload 2

# %%
import itertools
import math
import timeit
from decimal import Decimal

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.special as sp

# %%
from wolfram_grid import wolfram_dda, search_grid
from algorithms_stan import stan_grad_reg_inc_beta, stan_inc_beta, stan_inc_beta_strict
from algorithm_inbeder import inbeder as inbeder1
from algorithm_inbeder_documented import inbeder as inbeder2
from plot import plot_zlevel, plot_time

# %%
algos = dict(
    stan_grad_reg_inc_beta = stan_grad_reg_inc_beta,
    stan_inc_beta_strict = stan_inc_beta_strict, 
    inbeder1 = inbeder1,
    inbeder2 = inbeder2,
)

# %% [markdown]
# # Accuracy comparison

# %%
a = 15
b = 1.25
z = 0.999

# %%
x = np.linspace(0, 50, 10000)
y = sp.betainc(x, b, z)
plt.plot(x, y)
plt.axvline(a, c='k', ls='dashed', alpha=.5)
plt.ylabel('incbeta')
plt.xlabel('a')
plt.title(f'{z=}, {b=}');

# %%
print(f'{"wolfram".rjust(25)}: {wolfram_dda(z, a, b)}')
for algo_name, algo in algos.items():
    print(f'{algo_name.rjust(25)}: {algo(z, a, b)[0]}')


# %%
def safe_division(xs, ys):
    result = []
    for x, y in zip(xs, ys):
        if y != 0:
            result.append(x / y)
        else:
            result.append(np.nan)
    return np.array(result)

def safe_log(xs):
    result = []
    for x in xs:
        if x != 0:
            result.append(math.log(x))
        else:
            result.append(np.NINF)
            
    return np.array(result)


# %%
df = pd.DataFrame(
    columns = ('z', 'a', 'b', 'wolfram', *algos.keys())
)

for z, a, b in itertools.product(search_grid['z'], search_grid['a'], search_grid['b']):
    w_dda = Decimal(wolfram_dda(z, a, b))
    algos_dda = [Decimal(algo(z, a, b)[0]) for algo in algos.values()]
    df.loc[len(df)] = [z, a, b, w_dda, *algos_dda]

for algo_name in algos.keys():
    df[f'{algo_name}_abserr'] = np.abs(df['wolfram'] - df[algo_name])
for algo_name in algos.keys():
    df[f'{algo_name}_relerr'] = np.abs(1 - safe_division(df[algo_name], df['wolfram']))
for algo_name in algos.keys():
    df[f'{algo_name}_abserr_log'] = safe_log(df[f'{algo_name}_abserr'])
for algo_name in algos.keys():
    df[f'{algo_name}_relerr_log'] = safe_log(df[f'{algo_name}_relerr'])
    
# df.to_csv('dda.csv')

# %%
# df = pd.read_csv('dda.csv', index_col=0)
# df['z'] = df['z'].round(3)

# %%
df

# %%
for z in search_grid['z']:
    lnvalues = df.loc[df['z']==z, 'stan_grad_reg_inc_beta_abserr_log':'inbeder2_abserr_log'].values
    flnvalues = lnvalues[np.isfinite(lnvalues)]
    print(z, np.min(flnvalues), np.max(flnvalues))

# %%
for z in search_grid['z']:
    tdf = df[df['z']==z][[
        'z', 'a', 'b', 
        'stan_grad_reg_inc_beta_abserr_log', 
        'stan_inc_beta_strict_abserr_log',
        'inbeder1_abserr_log',
        'inbeder2_abserr_log',
    ]]
    plot_zlevel(tdf, vmin=-50, vmax=10)

# %%
for z in search_grid['z']:
    lnvalues = df.loc[df['z']==z, 'stan_grad_reg_inc_beta_relerr_log':'inbeder_relerr_log'].values
    flnvalues = lnvalues[np.isfinite(lnvalues)]
    print(z, np.min(flnvalues), np.max(flnvalues))

# %%
for z in search_grid['z']:
    tdf = df[df['z']==z][[
        'z', 'a', 'b', 
        'stan_grad_reg_inc_beta_relerr_log', 
        'stan_inc_beta_relerr_log',
        'stan_inc_beta_strict_relerr_log',
        'inbeder_relerr_log',
    ]]
    plot_zlevel(tdf, vmin=-50, vmax=10)

# %% [markdown]
# # Speed comparison

# %%
tdf = pd.DataFrame(
    columns = ('z', 'a', 'b', *algos.keys())
)

for i, (z, a, b) in enumerate(itertools.product(search_grid['z'], search_grid['a'], search_grid['b'])):
    if i % 5 == 0:
        print(i)

    algos_avg_time = []
    for j, algo in enumerate(algos.values()):
        if j <= 0:
            algos_avg_time.append(1)
            continue
        algo_avg_time = timeit.timeit(lambda: algo(z, a, b), number=100) / 100
        algos_avg_time.append(algo_avg_time)
    
    tdf.loc[len(tdf)] = [z, a, b, *algos_avg_time]

for algo_name in algos.keys():
    tdf[f'{algo_name}_log'] = np.log(tdf[algo_name])

# tdf.to_csv('time.csv')

# %%
# tdf = pd.read_csv('time.csv', index_col=0)
# tdf['z'] = tdf['z'].round(3)

# %%
tdf

# %%
for z in search_grid['z']:
    lnvalues = tdf.loc[tdf['z']==z, 'stan_grad_reg_inc_beta_log':'inbeder2_log'].values
    print(z, np.min(lnvalues), np.max(lnvalues))

# %%
for z in search_grid['z']:
    ttdf = tdf[tdf['z']==z][[
        'z', 'a', 'b', 
        'stan_grad_reg_inc_beta_log', 
        'stan_inc_beta_strict_log',
        'inbeder1_log',
        'inbeder2_log',
    ]]
    plot_time(ttdf, -11, 1)

# %% [markdown]
# ## Cases where wolfram and stan_grad_reg_inc_beta diverge from rest

# %% [markdown]
# ### Wolfram derivative is way off

# %%
z = 0.25
a = 150
b = 1250

# %%
x = np.linspace(0, max(a, b), 10000)
y = sp.betainc(x, b, z)
plt.plot(x, y)
plt.axvline(a, c='k', ls='dashed', alpha=.5)
plt.ylabel('incbeta')
plt.xlabel('a')
plt.title(f'{z=}, {b=}');

# %%
print(f'{"wolfram".rjust(25)}: {wolfram_dda(z, a, b)}')
for algo_name, algo in algos.items():
    print(f'{algo_name.rjust(25)}: {algo(z, a, b)[0]}')

# %% [markdown]
# ### Wolfram and stan_grad_reg_inc_beta agree, but it looks like derivative should be 0

# %%
z = 0.25
a = 15
b = 12500

# %%
x = np.linspace(0, max(a, b), 10000)
y = sp.betainc(x, b, z)
plt.plot(x, y)
plt.axvline(a, c='k', ls='dashed', alpha=.5)

plt.ylabel('incbeta')
plt.xlabel('a')
plt.title(f'{z=}, {b=}');

# %%
print(f'{"wolfram".rjust(25)}: {wolfram_dda(z, a, b)}')
for algo_name, algo in algos.items():
    print(f'{algo_name.rjust(25)}: {algo(z, a, b)[0]}')

# %% [markdown]
# ### All fail to converge, except inbeder which returns 0

# %%
z = 0.75
a = 15000
b = 12500

# %%
x = np.linspace(0, b * 8, 10000)
y = sp.betainc(x, b, z)
plt.plot(x, y)
plt.axvline(a, c='k', ls='dashed', alpha=.5)

plt.ylabel('incbeta')
plt.xlabel('a')
plt.title(f'{z=}, {b=}');

# %%
print(f'{"wolfram".rjust(25)}: {wolfram_dda(z, a, b)}')
for algo_name, algo in algos.items():
    print(f'{algo_name.rjust(25)}: {algo(z, a, b)[0]}')

# %%
