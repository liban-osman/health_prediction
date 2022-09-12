import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Health-Prediction/data/train.csv")

TARGET = 'health'
col_names = [col for col in df.columns]

# take two columns (i.e. their names)
# find rows where both are not nan
# return resulting columns as np arrays
def remove_joint_nans(x, y):
    x = np.array(df[x])
    y = np.array(df[y])

    x_not_nan = np.logical_not(np.isnan(x))
    y_not_nan = np.logical_not(np.isnan(y))
    both_not_nan = np.logical_and(x_not_nan, y_not_nan)
    
    inds = np.where(both_not_nan)[0] # inds of all non nan elements
    print("{} values after removing NaNs from x and y".format(len(inds)))
    
    x = x[inds]
    y = y[inds]

    return x, y

# Same as above but single case
def remove_nans(x):
    x = np.array(df[x])

    x_not_nan = np.logical_not(np.isnan(x))
    
    inds = np.where(x_not_nan)[0] # inds of all non nan elements
    print("{} values after removing NaNs from x, {:.4f}%".format(len(inds), 100 * len(inds)/len(x)))
    
    x = x[inds]

    return x

# takes column name, draws histogram
# optional scale function (i.e. np.log, np.exp, np.sqrt)
def col_hist(x, scale = None):
    x = remove_nans(x)
    if scale is not None:
        x = scale(x)
    
    plt.hist(x)
    plt.show()
    plt.close()

# Take list of x's and hist all of them
import math
def col_hist_subplots(x):
    x = [remove_nans(x) for x in x]

    n = len(x)
    per_row = 8
    n_rows = math.ceil(n / per_row)

    fig, axs = plt.subplots(n_rows, per_row)
    i = 0
    for r in range(n_rows):
        for c in range(per_row):
            axs[r,c].hist(x[i])
            i+=1
            if i >= n: break
    plt.show()
    plt.close()

# takes column names, draws scatterplot
def col_scatter(x, y = TARGET, scale_x = None, scale_y = None):
    x_name = x
    y_name = y
    x, y = remove_joint_nans(x, y)

    if scale_x is not None:
        x = scale_x(x)
    if scale_y is not None:
        y = scale_y(y)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    plt.scatter(x, y)
    
    plt.show()
    plt.close()

def col_scatter_subplots(X, Y = TARGET):

    n = len(X)
    per_row = 8
    n_rows = math.ceil(n / per_row)

    fig, axs = plt.subplots(n_rows, per_row)
    i = 0
    y_name = Y
    for r in range(n_rows):
        for c in range(per_row):
            x_name = X[i]
            x, y = remove_joint_nans(X[i], Y)

            axs[r,c].set_xlabel(x_name)
            axs[r,c].set_ylabel(y_name)
            axs[r,c].scatter(x, y)
            i+=1
            if i >= n: break
    plt.show()
    plt.close()

from utils import apply_jet
# For categorical variables the above function is useless
# Makes more sense to visualize a confusion plot

def hist2dhelper(x, y):
    d = {} # dict of (pair, freq) pairs

    for (x_i, y_i) in zip(x, y):
        if (x_i, y_i) in d:
            d[(x_i, y_i)] += 1
        else:
            d[(x_i, y_i)] = 1

    n = len(x)
    x = []
    y = []
    z = []
    
    for (x_i, y_i) in d:
        x.append(x_i)
        y.append(y_i)
        z.append(d[(x_i, y_i)])
        
    x = np.array(x)
    y = np.array(y)
    z = np.array(z) / n # proportion
    colors = apply_jet(z)

    return x, y, colors

def hist2d(x, y = TARGET, scale_x = None, scale_y = None):
    x_name = x
    y_name = y
    x, y = remove_joint_nans(x, y)

    if scale_x is not None:
        x = scale_x(x)
    if scale_y is not None:
        y = scale_y(y)

    x, y, colors = hist2dhelper(x, y)
    
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.scatter(x, y, s = 3000, c = colors, marker = 's')

    plt.show()
    plt.close()

def hist2d_subplots(X, Y = TARGET):
    y_name = Y
    n = len(X)
    per_row = 8
    n_rows = math.ceil(n / per_row)

    fig, axs = plt.subplots(n_rows, per_row)
    i = 0
    y_name = Y
    for r in range(n_rows):
        for c in range(per_row):
            x_name = X[i]
            x, y = remove_joint_nans(X[i], Y)
            x, y, colors = hist2dhelper(x, y)

            axs[r,c].set_xlabel(x_name)
            axs[r,c].set_ylabel(y_name)
            axs[r,c].scatter(x, y, s = 100, c = colors, marker = 's')
            i+=1
            if i >= n: break
    plt.show()
    plt.close()

    
# Get NaN counts per column and plot them
# set prop = True to get proportions instead
def all_nan_counts(prop = True):
    X = []
    Y = []
    for i, col in enumerate(col_names):
        x = np.array(df[col])
        inds = np.where(np.isnan(x))[0] # inds of all nans
        nan_count = len(inds)

        X.append(i)
        Y.append(nan_count/len(x) if prop else nan_count)

    plt.plot(X, Y)
    plt.show()
    plt.close()


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Checks the VIF between columns and removes highly correlated columns
# REQUIRES no NaN's and infinity values
def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

if __name__ == "__main__":
    relevant_vars = ['x472', 'x545', 'x597', 'x632', 'x635', 'x641', 'x642', 'x643', 'x644',
       'x645', 'x646', 'x647', 'x648', 'x649', 'x650', 'x651', 'x652', 'x657',
       'x754', 'x893', 'x898', 'x902', 'x906', 'x907', 'x908', 'x909', 'x934',
       'x935', 'x939', 'x940', 'x941', 'x942', 'x1032', 'x1035', 'x1036',
       'x1143', 'x1147', 'x1150', 'x1159', 'x1160', 'x1161', 'x1162', 'x1181',
       'x1184', 'x1185']

    categoricals = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39,
                    40, 41, 43]
    continuous = [3,18, 32, 33, 34, 35, 36, 37, 42, 44]

    categorical = [relevant_vars[ind] for ind in categoricals]
    continuous = [relevant_vars[ind] for ind in continuous]

    print("Categoricals: {}".format(categorical))
    print("Continuous: {}".format(continuous))
    
    #col_scatter_subplots(continuous)
    hist2d_subplots(categorical)
