import numpy as np

# get a score for whether y_pred
# is simply outputting one class
# specifically looks at misclassifications
def output_stddev(y_pred, y_true):
    misclass = y_pred[y_pred == y_true]
    return misclass.std()

def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / len(y_true)

# get frequencies of each y value
# turns it into a distribution
def get_freqs(y):
    y_min = y.min()
    y_max = y.max()
    y_vals = [i for i in range(y_min, y_max + 1)]
    freqs = [(y == val).sum() for val in y_vals]
    freqs = np.array(freqs) 
    return freqs / freqs.sum()

# baseline accuracies for certain metrics:
# 1. predicting randomly
# 2. predicting modal class
def generate_baselines(y_train, y_test):
    rand_trials = 100
    accs = []
    for i in range(rand_trials):
        preds = np.random.randint(0, y_train.max(), (len(y_test),))
        accs.append(accuracy(preds, y_test))
    rand_acc = sum(accs) / len(accs)

    freqs = get_freqs(y_train)
    modal_label = freqs.argmax() + 1

    modal_acc = accuracy(np.array([modal_label] * len(y_test)), y_test)

    return rand_acc, modal_acc


