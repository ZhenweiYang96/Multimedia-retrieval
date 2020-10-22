import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_to_float(value):
    col = value.str.strip('[]')
    col = col.str.split(',')
    for i in range(0, len(col)):
        col[i] = [float(j) for j in col[i]]

    return col

df = pd.read_csv('excel_file\\experimental.csv')
df[['D4_500k', 'D4_1mil', 'D4_2mil', 'D4_5mil']] = df[['D4_500k', 'D4_1mil', 'D4_2mil', 'D4_5mil']].apply(convert_to_float)
df = df[['D4_500k', 'D4_1mil', 'D4_2mil', 'D4_5mil']]






def plot_dis_single(df, row, col_start, col_end):
    fig, ax = plt.subplots(row, col_end)
    a = 0
    b = 0
    for i in range(0, row):
        print('i', i)
        for j in range(col_start, col_end):
            sns.kdeplot(df.iat[i, j], ax=ax[a, b])
            if b == col_end - 1:
                b = 0
                a += 1
            else:
                b += 1
    plt.show()


#plot_dis_single(df, 5, 0, 4)
print('ok')

def plot_dist_merge(df):
    fig, ax = plt.subplots(1, 4)
    a = 0
    b = 0
    #col
    for i in range(0, 4):
        #row
        for j in range(0, 10):
            sns.kdeplot(df.iat[j, i], ax=ax[b])
        b += 1
    plt.show()


plot_dist_merge(df)
