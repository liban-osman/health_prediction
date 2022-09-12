from pt_data_utils  import UnevenDataLoader, StandardDataLoader
from pt_data_utils import prepare_PT_input
from data_subset import get_split_inds

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 5

# returns  trained model, train losses, test accuracies, standard deviations of incorrect predictions
def train_PT(X_train, X_test, y_train, y_test):
    y_train = torch.from_numpy(y_train) - 1 # (-1 forces classes to be [0, 4])
    y_train = y_train.long()
    
    X_train_cat, X_train_regr = prepare_PT_input(X_train)
    
    # Want to turn each categorical variable into one hot vectors
    # To stay 0 centered, use -1 in place of 0
    class OneHotEmbedding:
        def __init__(self, min_, max_):
            self.n_vals = max_ - min_ + 1
        
        def forward(self, x):
            # x is [n,] categories
            n = x.shape[0]
            y = torch.ones(n, self.n_vals, device = device) * -1
            y[torch.arange(n), x] = 1
            return y
    
    # takes tensor of categoricals (LongTensor) T shape [n,]
    # makes corresponding one hot layer
    def get_onehot(T):
        max_val = round(T.max().item())
        min_val = round(T.min().item())
        return OneHotEmbedding(min_val, max_val)
    
    # The actual model
    class Net(nn.Module):
        def __init__(self, n_layers, n_nodes, out_classes):
            super().__init__()
            
            # Store these for later
            cat_inds, regr_inds = get_split_inds() 
                
            # Make one hot layers
            self.embedding_layers = []
            for i in range(len(cat_inds)):
                self.embedding_layers.append(get_onehot(X_train_cat[:,i]))
            
            # need to get dimensionality of all one hots together
            dim = 0
            for layer in self.embedding_layers:
                dim += layer.n_vals
            
            # add dimension for each continuous variable
            dim += len(regr_inds)

            self.layers = nn.Sequential(*
                [nn.Linear(dim, n_nodes), nn.LeakyReLU()] + \
                [nn.Linear(n_nodes, n_nodes), nn.LeakyReLU(), nn.Dropout(0)] * (n_layers - 1) + \
                [nn.Linear(n_nodes, out_classes)]
            )
        
        def prep_input(self, x_cats, x_regr):
            one_hots = []
            for i, layer in enumerate(self.embedding_layers):
                one_hots.append(layer.forward(x_cats[:,i])) # -> each [n, dim_i]
            one_hots = torch.cat(one_hots, dim = 1)

            # x_regr: [n, len(regr_inds)]
            x = torch.cat([one_hots.to(device),  x_regr.to(device)], dim = 1) # ->  [n, dim]
            return x
            
        def forward(self, x):  
            logits = self.layers(x)
            return logits

        # This assumes input is basic data matrix as np array
        # this is to simplify using this downstream
        # x: [n, feat]
        def classify(self, x):
            with torch.no_grad():
                x_cat, x_regr = prepare_PT_input(x)
                x_cat = x_cat.to(device)
                x_regr = x_regr.to(device)
                x = self.prep_input(x_cat, x_regr)

                logits = self.forward(x)
                return logits.argmax(1).cpu().numpy() # return labels 
        

    
    # test the model on np arrays X and y
    def test(model, X, y):
        y_pred = model.classify(X)
        acc = (y_pred == y).sum() / len(y)
        mc_std = metrics.output_stddev(y_pred, y)
        return acc, mc_std
        
    # Now for actual training
    
    #loader = UnevenDataLoader(X_train_cat, X_train_regr, y_train)
    loader = StandardDataLoader(X_train_cat, X_train_regr, y_train)
    
    model = Net(3, 513, n_classes)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 0.1)
    
    n_iterations = 100
    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    accs = []
    mc_stds = []

    for i in range(n_iterations):
        x1, x2, y = loader.sample(batch_size)
        with torch.no_grad():
            x = model.prep_input(x1, x2)
        
        x = x.to(device)
        y = y.to(device)
    
        opt.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)
        
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        acc, mc_std = test(model, X_test, y_test)
        accs.append(acc)
        mc_stds.append(mc_std)
        
        print("Iteration [{}/{}], Loss: {}, Test Accuracy: {}".format(i, n_iterations, loss.item(), acc))
    
    return model, losses, accs, mc_stds

def train_PT_nocats(X_train, X_test, y_train, y_test):
    y_train = torch.from_numpy(y_train) - 1 # (-1 forces classes to be [0, 4])
    y_train = y_train.long()
    
    X_train_cat, X_train_regr = prepare_PT_input(X_train)
    X_train = X_train_regr
    
    # The actual model
    class Net(nn.Module):
        def __init__(self, n_layers, n_nodes, out_classes):
            super().__init__()
            
            # Store these for later
            cat_inds, regr_inds = get_split_inds() 
                
            # add dimension for each continuous variable
            dim = len(regr_inds)

            self.layers = nn.Sequential(*
                [nn.Linear(dim, n_nodes), nn.LeakyReLU()] + \
                [nn.Linear(n_nodes, n_nodes), nn.LeakyReLU(), nn.Dropout(0)] * (n_layers - 1) + \
                [nn.Linear(n_nodes, out_classes)]
            )
        
        def forward(self, x):  
            logits = self.layers(x)
            return logits

        # This assumes input is basic data matrix as np array
        # this is to simplify using this downstream
        # x: [n, feat]
        def classify(self, x):
            with torch.no_grad():
                x_cat, x_regr = prepare_PT_input(x)
                x = x_regr.to(device)
                logits = self.forward(x)
                return logits.argmax(1).cpu().numpy() # return labels 
        

    
    # test the model on np arrays X and y
    def test(model, X, y):
        y_pred = model.classify(X)
        acc = (y_pred == y).sum() / len(y)
        mc_std = metrics.output_stddev(y_pred, y)
        return acc, mc_std
        
    # Now for actual training
    loader = StandardDataLoader(X_train, X_train, y_train)
    
    model = Net(3, 1024, n_classes)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 0.1)
    
    n_iterations = 100
    batch_size = 64

    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    accs = []
    mc_stds = []

    for i in range(n_iterations):
        x1, x2, y = loader.sample(batch_size)
        x = x2
        x = x.to(device)
        y = y.to(device)
    
        opt.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)
        
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        acc, mc_std = test(model, X_test, y_test)
        accs.append(acc)
        mc_stds.append(mc_std)
        
        print("Iteration [{}/{}], Loss: {}, Test Accuracy: {}".format(i, n_iterations, loss.item(), acc))
    
    return model, losses, accs, mc_stds