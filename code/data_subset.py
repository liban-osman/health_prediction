# Extracts relevant subset of data to a numpy matrix

relevant_feature_names = ['x472', 'x545', 'x597', 'x632', 'x635', 'x641', 'x642', 'x643', 'x644',
       'x645', 'x646', 'x647', 'x648', 'x649', 'x650', 'x651', 'x652', 'x657',
       'x754', 'x893', 'x898', 'x902', 'x906', 'x907', 'x908', 'x909', 'x934',
       'x935', 'x939', 'x940', 'x941', 'x942', 'x1032', 'x1035', 'x1036',
       'x1143', 'x1147', 'x1150', 'x1159', 'x1160', 'x1161', 'x1162', 'x1181',
       'x1184', 'x1185']

# get indices of all columns that are categorical variables
# takes list of feature names (i.e. ['x472','x545'])
def get_cat_inds(L):
    # note for later: took out x635 since it doesn't look categorical
    cats = ['x472', 'x545', 'x597', 'x641', 'x642', 'x643', 'x644', 'x645', 'x646', 'x647', 'x648', 'x649', 'x650', 'x651', 'x652', 'x657', 'x893', 'x898', 'x902', 'x906', 'x907', 'x908', 'x909', 'x934', 'x935', 'x939', 'x940', 'x941', 'x942', 'x1159', 'x1160', 'x1161', 'x1162', 'x1184']
    inds = []
    for i, x in enumerate(L):
        if x in cats:
            inds.append(i)
    return inds

# get inds for categoricals and continous
def get_split_inds(L = relevant_feature_names):
    n = len(L)
    cat_inds = get_cat_inds(L)
    regr_inds = set([i for i in range(n)]) - set(cat_inds)
    regr_inds  = list(regr_inds)
    return cat_inds, regr_inds