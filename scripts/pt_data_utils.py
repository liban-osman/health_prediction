import torch
from data_subset import get_split_inds

# Utilities for converting np data to pytorch format

# Prepare batched input for pytorch model
# assumes shape: [n, feat]
def prepare_PT_input(X):
    cat_inds, regr_inds = get_split_inds() 
    X = torch.from_numpy(X)
    
    # verify categoricals can be ints
    def verify_cats():
        for ind in cat_inds:
            X_cat = X[:, ind]
            X_cat_rounded = torch.round(X_cat)
            if (X_cat - X_cat_rounded).mean() > 1e-6:
                print("{} not categorical!".format(ind))
                assert False
    
    verify_cats()
    
    X_cat = X[:, cat_inds].long()
    
    # For the categoricals, need the ones in levels to become 0 if their min isnt 0
    for i in range(len(cat_inds)):
        min_val = X_cat[:,i].min().item()
        if min_val != 0: # means it must be [min_val, ...]
            X_cat[:,i] -= min_val # now [0, ...]
            
    X_regr = X[:, regr_inds].float()
    
    def normalize_col(col):
        maxi = col.max()
        mini = col.min()
        
        col -= mini # [0, ...]
        col /= (maxi/2) # [0, 2]
        col -= 1 # [-1, 1]
        return col
        
    for i in range(X_regr.shape[1]):
        X_regr[:, i] = normalize_col(X_regr[:, i])
    
    return X_cat, X_regr

# forces all y instances to be weighted equally
class UnevenDataLoader:
    # assumes each y instance can be made into int
    def __init__(self, X_cat, X_regr, y):
        self.n = len(X_cat)
        assert (self.n == len(X_regr)) and (self.n == len(y))

        self.n_classes = 5

        # Store dictionary for each output type
        self.dict = {}

        # Initialize each y_i as having two empty lists
        for i in range(self.n_classes):
            self.dict[i] = [[],[]]

        # add respective features to their label lists
        for i in range(self.n):
            y_i = int(y[i].item())
            self.dict[y_i][0].append(X_cat[i])
            self.dict[y_i][1].append(X_regr[i])

        # stack so they're all tensors rather than lists
        for i in range(self.n_classes):
            self.dict[i][0] = torch.stack(self.dict[i][0]) # -> [n_i, cat_features]
            self.dict[i][1] = torch.stack(self.dict[i][1]) # -> [n_i, cont_features]

    # randomly sample n samples from y = i
    def random_sample(self, n, i):
        n_i = len(self.dict[i][0])
        inds = torch.randint(n_i, (n,))
        return self.dict[i][0][inds], self.dict[i][1][inds]

    # randomly sample n samples from all possible y
    def sample(self, n):
        # y's to use for each sample instance
        y_vals = torch.randint(self.n_classes, (n,))
        # number of samples from each y 
        n_ys = [(y_vals==i).sum().item() for i in range(self.n_classes)]

        X_cat = []
        X_regr  = []

        for i in range(self.n_classes):
            if n_ys[i] == 0:
                continue
            cat, regr = self.random_sample(n_ys[i], i)
            X_cat.append(cat)
            X_regr.append(regr)

        return torch.cat(X_cat), torch.cat(X_regr), y_vals

class StandardDataLoader:
    def __init__(self, X_cat, X_regr, y):
        self.n = len(X_cat)
        assert self.n == len(X_regr) and self.n == len(y)

        self.X_1 = X_cat
        self.X_2 = X_regr
        self.y = y
    
    def sample(self, n):
        inds = torch.randint(self.n, (n,))
        return self.X_1[inds], self.X_2[inds], self.y[inds]