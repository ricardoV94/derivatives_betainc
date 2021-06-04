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
import time
import itertools
import math
import timeit
from decimal import Decimal
import warnings

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import aesara
import aesara.tensor as at

# %%
from wolfram_grid import wolfram_dda, search_grid
from aesara_betainc import betainc
from aesara_incomplete_beta import incomplete_beta
from plot import plot_zlevel, plot_time

# %% [markdown]
# # Compile functions

# %% [markdown]
# ## incomplete_beta

# %%
# Compile function
z = at.scalar('z')
a = at.scalar('a')
b = at.scalar('b')

out = incomplete_beta(a, b, z)
dda = aesara.grad(out, a)
ddb = aesara.grad(out, b)

# %%
start_compilation = time.time()
incomplete_beta_fn = aesara.function([z, a, b], out)
end_compilation = time.time()
print(end_compilation - start_compilation)

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    start_compilation = time.time()
    incomplete_beta_der = aesara.function([z, a, b], [dda, ddb])
    end_compilation = time.time()
    print(end_compilation - start_compilation)

# %% [markdown]
# ## betainc

# %%
# Compile function
z = at.scalar('z')
a = at.scalar('a')
b = at.scalar('b')

out = betainc(a, b, z)
dda = aesara.grad(out, a)
ddb = aesara.grad(out, b)

# %%
start_compilation = time.time()
betainc_fn = aesara.function([z, a, b], out)
end_compilation = time.time()
print(end_compilation - start_compilation)

# %%
start_compilation = time.time()
betainc_der = aesara.function([z, a, b], [dda, ddb])
end_compilation = time.time()
print(end_compilation - start_compilation)

# %% [markdown]
# # Op comparison

# %%
algos = dict(
    incomplete_beta = incomplete_beta_fn,
    betainc = betainc_fn
)

# %% [markdown]
# ## Accuracy

# %%
a = 15
b = 1.25
z = 0.999

# %%
print(f'{"scipy".rjust(25)}: {sp.betainc(a, b, z)}')
for algo_name, algo in algos.items():
    print(f'{algo_name.rjust(25)}: {algo(z, a, b)}')


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
    columns = ('z', 'a', 'b', 'scipy', *algos.keys())
)

for z, a, b in itertools.product(search_grid['z'], search_grid['a'], search_grid['b']):
    scipy_out = Decimal(sp.betainc(a, b, z))
    algos_out = [Decimal(float(algo(z, a, b))) for algo in algos.values()]
    df.loc[len(df)] = [z, a, b, scipy_out, *algos_out]

for algo_name in algos.keys():
    df[f'{algo_name}_abserr'] = np.abs(df['scipy'] - df[algo_name])
for algo_name in algos.keys():
    df[f'{algo_name}_relerr'] = np.abs(1 - safe_division(df[algo_name], df['scipy']))
for algo_name in algos.keys():
    df[f'{algo_name}_abserr_log'] = safe_log(df[f'{algo_name}_abserr'])
for algo_name in algos.keys():
    df[f'{algo_name}_relerr_log'] = safe_log(df[f'{algo_name}_relerr'])

# %%
df

# %%
for z in search_grid['z']:
    lnvalues = df.loc[df['z']==z, 'incomplete_beta_abserr_log':'betainc_abserr_log'].values
    flnvalues = lnvalues[np.isfinite(lnvalues)]
    print(z, np.min(flnvalues), np.max(flnvalues))

# %%
for z in search_grid['z']:
    tdf = df[df['z']==z][[
        'z', 'a', 'b', 
        'incomplete_beta_abserr_log', 
        'betainc_abserr_log',
    ]]
    plot_zlevel(tdf, vmin=-50, vmax=10)

# %%
for z in search_grid['z']:
    lnvalues = df.loc[df['z']==z, 'incomplete_beta_relerr_log':'betainc_relerr_log'].values
    flnvalues = lnvalues[np.isfinite(lnvalues)]
    print(z, np.min(flnvalues), np.max(flnvalues))

# %%
for z in search_grid['z']:
    tdf = df[df['z']==z][[
        'z', 'a', 'b', 
        'incomplete_beta_relerr_log', 
        'betainc_relerr_log',
    ]]
    plot_zlevel(tdf, vmin=-50, vmax=10)

# %% [markdown]
# ## Speed

# %%
tdf = pd.DataFrame(
    columns = ('z', 'a', 'b', *algos.keys())
)

for i, (z, a, b) in enumerate(itertools.product(search_grid['z'], search_grid['a'], search_grid['b'])):
    if i % 5 == 0:
        print(i)

    algos_avg_time = []
    for j, algo in enumerate(algos.values()):
        algo_avg_time = timeit.timeit(lambda: algo(z, a, b), number=100) / 100
        algos_avg_time.append(algo_avg_time)
    
    tdf.loc[len(tdf)] = [z, a, b, *algos_avg_time]

for algo_name in algos.keys():
    tdf[f'{algo_name}_log'] = np.log(tdf[algo_name])

# %%
tdf

# %%
for z in search_grid['z']:
    ttdf = tdf[tdf['z']==z][[
        'z', 'a', 'b', 
        'incomplete_beta_log', 
        'betainc_log',
    ]]
    plot_time(ttdf, -11, 1)

# %% [markdown]
# # Derivative comparison

# %%
algos = dict(
    incomplete_beta = incomplete_beta_der,
    betainc = betainc_der
)

# %% [markdown]
# ## Accuracy

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
df = pd.DataFrame(
    columns = ('z', 'a', 'b', 'wolfram', *algos.keys())
)

for z, a, b in itertools.product(search_grid['z'], search_grid['a'], search_grid['b']):
    w_dda = Decimal(wolfram_dda(z, a, b))
    algos_dda = [Decimal(float(algo(z, a, b)[0])) for algo in algos.values()]
    df.loc[len(df)] = [z, a, b, w_dda, *algos_dda]

for algo_name in algos.keys():
    df[f'{algo_name}_abserr'] = np.abs(df['wolfram'] - df[algo_name])
for algo_name in algos.keys():
    df[f'{algo_name}_relerr'] = np.abs(1 - safe_division(df[algo_name], df['wolfram']))
for algo_name in algos.keys():
    df[f'{algo_name}_abserr_log'] = safe_log(df[f'{algo_name}_abserr'])
for algo_name in algos.keys():
    df[f'{algo_name}_relerr_log'] = safe_log(df[f'{algo_name}_relerr'])

# %%
df

# %%
for z in search_grid['z']:
    lnvalues = df.loc[df['z']==z, 'incomplete_beta_abserr_log':'betainc_abserr_log'].values
    flnvalues = lnvalues[np.isfinite(lnvalues)]
    print(z, np.min(flnvalues), np.max(flnvalues))

# %%
for z in search_grid['z']:
    tdf = df[df['z']==z][[
        'z', 'a', 'b', 
        'incomplete_beta_abserr_log', 
        'betainc_abserr_log',
    ]]
    plot_zlevel(tdf, vmin=-50, vmax=10)

# %%
for z in search_grid['z']:
    lnvalues = df.loc[df['z']==z, 'incomplete_beta_relerr_log':'betainc_relerr_log'].values
    flnvalues = lnvalues[np.isfinite(lnvalues)]
    print(z, np.min(flnvalues), np.max(flnvalues))

# %%
for z in search_grid['z']:
    tdf = df[df['z']==z][[
        'z', 'a', 'b', 
        'incomplete_beta_relerr_log', 
        'betainc_relerr_log',
    ]]
    plot_zlevel(tdf, vmin=-50, vmax=10)

# %% [markdown]
# ## Speed

# %%
tdf = pd.DataFrame(
    columns = ('z', 'a', 'b', *algos.keys())
)

for i, (z, a, b) in enumerate(itertools.product(search_grid['z'], search_grid['a'], search_grid['b'])):
    if i % 5 == 0:
        print(i)

    algos_avg_time = []
    for j, algo in enumerate(algos.values()):
        algo_avg_time = timeit.timeit(lambda: algo(z, a, b), number=100) / 100
        algos_avg_time.append(algo_avg_time)
    
    tdf.loc[len(tdf)] = [z, a, b, *algos_avg_time]

for algo_name in algos.keys():
    tdf[f'{algo_name}_log'] = np.log(tdf[algo_name])

# %%
tdf

# %%
for z in search_grid['z']:
    ttdf = tdf[tdf['z']==z][[
        'z', 'a', 'b', 
        'incomplete_beta_log', 
        'betainc_log',
    ]]
    plot_time(ttdf, -11, 1)

# %%
