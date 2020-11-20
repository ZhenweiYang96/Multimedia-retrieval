import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


def plot_distribution(df, col, xmin, xmax, limit):
    fig, ax = plt.subplots(5, 4)
    plt.subplots_adjust(wspace=0.5, hspace=1)
    a = 0
    b = 0
    for i in range(0, 19):
        print(i)
        for j in range(i * 20, i * 20 + 20):
            sns.kdeplot(df.loc[j, col], ax=ax[a, b]).set(title=df.loc[j, 'class'], xlim=(xmin, xmax), ylim=(0, limit))
        if b == 3:
            b = 0
            a += 1
        else:
            b += 1
    plt.show()


def convert_to_float(value):
    col = value.str.strip('[]')
    col = col.str.split(',')
    for i in range(0, len(col)):
        col[i] = [float(j) for j in col[i]]

    return col


###Normalize the columns so mean = 0 and std = 1
def standardize(col):
    col = col.apply(lambda x: np.array(x))
    return (col - col.map(lambda x: x.mean()))/col.map(lambda x: x.std())


def to_bin(df, plot=False):

    df['A3'] = standardize(df['A3'])
    df['D1'] = standardize(df['D1'])
    df['D2'] = standardize(df['D2'])
    df['D3'] = standardize(df['D3'])
    df['D4'] = standardize(df['D4'])

    ###Get minimum and maximum value of the distributions, used for histogram
    a3min = min(df['A3'].map(lambda x: x.min()))
    d1min = min(df['D1'].map(lambda x: x.min()))
    d2min = min(df['D2'].map(lambda x: x.min()))
    d3min = min(df['D3'].map(lambda x: x.min()))
    d4min = min(df['D4'].map(lambda x: x.min()))
    a3max = max(df['A3'].map(lambda x: x.max()))
    d1max = max(df['D1'].map(lambda x: x.max()))
    d2max = max(df['D2'].map(lambda x: x.max()))
    d3max = max(df['D3'].map(lambda x: x.max()))
    d4max = max(df['D4'].map(lambda x: x.max()))

    if plot:
        print('A3')
        #plot_distribution(df, 'A3', a3min, a3max, 1.5)
        print('D1')
        #plot_distribution(df, 'D1', d1min, d1max, 1)
        print('D2')
        #plot_distribution(df, 'D2', d2min, d2max, 1)
        print('D3')
        #plot_distribution(df, 'D3', d3min, d3max, 1)
        print('D4')
        plot_distribution(df, 'D4', d4min, d4max, 1)
        print('end')
    df['A3'] = df['A3'].map(lambda x: np.histogram(x, bins=20, range=(a3min, a3max)))
    df['D1'] = df['D1'].map(lambda x: np.histogram(x, bins=20, range=(d1min, d1max)))
    df['D2'] = df['D2'].map(lambda x: np.histogram(x, bins=20, range=(d2min, d2max)))
    df['D3'] = df['D3'].map(lambda x: np.histogram(x, bins=20, range=(d3min, d3max)))
    df['D4'] = df['D4'].map(lambda x: np.histogram(x, bins=20, range=(d4min, d4max)))

    df['A3'] = [i[0] for i in df['A3']]
    df['D1'] = [i[0] for i in df['D1']]
    df['D2'] = [i[0] for i in df['D2']]
    df['D3'] = [i[0] for i in df['D3']]
    df['D4'] = [i[0] for i in df['D4']]
    return df


