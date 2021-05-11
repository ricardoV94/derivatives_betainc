import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_zlevel(df, vmin, vmax):

    fig, ax = plt.subplots(1, 4, figsize=(22, 4))

    z = df['z'].values
    assert(len(np.unique(z)) == 1)
    z = z[0]

    a = df['a'].values.reshape(5, 5)
    b = df['b'].values.reshape(5, 5)

    for i, axi in enumerate(ax):
        tdf = df.iloc[:, 3+i]
        val = tdf.values.reshape(5, 5)
        val[val == -np.inf] = -999
        val[val == np.inf] = 999
        cb = axi.imshow(val.T, vmin=vmin, vmax=vmax, cmap=cm.viridis)
        
        # plot result text
        for x in range(5):
            for y in range(5):
                v = val[x, y]
            
                v_str = f'{v:.1f}'
                if v == -999:
                    v_str = '-inf'
                elif v == 999:
                    v_str = 'inf'
            
                axi.text(x, y, v_str, ha='center', va='center')

        axi.set_title(tdf.name)

        plt.colorbar(cb, ax=axi)

    unique_a = np.unique(a)
    x_ticks = np.arange(len(unique_a))
    x_ticklabels = list(sorted(unique_a))

    unique_b = np.unique(b)
    y_ticks = np.arange(len(unique_b))
    y_ticklabels = list(sorted(unique_b))

    for axi in ax:
        axi.set_xlabel('a')
        axi.set_xticks(x_ticks)
        axi.set_xticklabels(x_ticklabels)

        axi.set_ylabel('b')
        axi.set_yticks(y_ticks)
        axi.set_yticklabels(y_ticklabels)

    fig.suptitle(f'z={z}', y=1)


def plot_time(df, vmin, vmax):

    fig, ax = plt.subplots(1, 4, figsize=(22, 4))

    z = df['z'].values
    assert(len(np.unique(z)) == 1)
    z = z[0]

    a = df['a'].values.reshape(5, 5)
    b = df['b'].values.reshape(5, 5)


    for i, axi in enumerate(ax, 0):
        tdf = df.iloc[:, 3+i]
        val = tdf.values.reshape(5, 5)
        val[val == -np.inf] = -999
        val[val == np.inf] = 999
        cb = axi.imshow(val.T, vmin=vmin, vmax=vmax, cmap=cm.viridis)
        
        # plot result text
        for x in range(5):
            for y in range(5):
                v = val[x, y]            
                v_str = f'{v:.1f}'
            
                axi.text(x, y, v_str, ha='center', va='center')

        axi.set_title(f'{tdf.name} (time)')

        plt.colorbar(cb, ax=axi)

    unique_a = np.unique(a)
    x_ticks = np.arange(len(unique_a))
    x_ticklabels = list(sorted(unique_a))

    unique_b = np.unique(b)
    y_ticks = np.arange(len(unique_b))
    y_ticklabels = list(sorted(unique_b))

    for axi in ax:
        axi.set_xlabel('a')
        axi.set_xticks(x_ticks)
        axi.set_xticklabels(x_ticklabels)

        axi.set_ylabel('b')
        axi.set_yticks(y_ticks)
        axi.set_yticklabels(y_ticklabels)

    fig.suptitle(f'z={z}', y=1)