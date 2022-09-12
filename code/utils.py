import numpy as np

# applies plt jet color map to some data
def apply_jet(x):
    x = 255 * (x - x.min()) / x.max()
    
    r_left = (1 / 80) * (x - 89)
    r_right = (-1 / 56) * (x - 227) + 1
    r = np.where(x <= 200, r_left, r_right)
    r = np.clip(r, 0, 1)

    g_left = (1 / 64) * (x - 32)
    g_right = (-1 / 69) * (x - 163) + 1
    g = np.where(x <= 130, g_left, g_right)
    g = np.clip(g, 0, 1)

    b_left = (1 / 56) * x + 0.5
    b_right = (-1 / 80) * (x - 86) + 1
    b = np.where(x <= 58, b_left, b_right)
    b = np.clip(b, 0, 1)

    y = np.concatenate((r[:, None], g[:, None], b[:, None]), axis = 1)
    return y

# Gets row indices for a new dataframe
def get_resample_inds(labels):
    n = len(labels)
    n_labels = int(labels.max())
    freqs = np.rint(get_freqs(labels) * n)
    freqs = freqs.astype(int)
    
    # want relative frequency to be 1/n
    # to do this without removing any samples
    # get count of modal label
    modal_count = freqs.max()
    model_ind = freqs.argmax()
    
    # get indices
    
    resamples_needed = [modal_count - freq for freq in freqs]
    resamples = [None] * n_labels
    
    # row inds for samples of each class
    inds = [None] * n_labels
    for i in range(n_labels):
        inds[i] = np.where(labels == i + 1)[0]
    
    # row indices for all of our resamples
    sample_inds_full = []
    for i in range(n_labels):
        y_i = labels[labels==i+1]
        n_i = resamples_needed[i]
        if n_i ==  0:
            continue
        # get n samples for this label
        sample_inds = np.random.randint(0, len(inds[i]), (n_i,)) # this indexes inds[i]
        sample_inds = inds[i][sample_inds] # now we have actual row indices for n resamples
        sample_inds_full.append(sample_inds) 
    
    sample_inds_full = np.concatenate(sample_inds_full) # one tensor of all resampled row indices
    np.random.shuffle(sample_inds_full) # random order
    
    # concatenate with row indices of rows already in dataset
    row_inds = np.concatenate((np.arange(n), sample_inds_full))
    
    return row_inds