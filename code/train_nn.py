import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Health-Prediction/data/train.csv')
df_missing=df.isna()
result = df.isna().median()
df = df.loc[:,result < .3]
df = df.fillna(df.median())

y = df["health"].copy()

X = df.drop(columns = ["health", "uniqueid", "personid"])
X_new = X[['x472', 'x545', 'x597', 'x632', 'x635', 'x641', 'x642', 'x643', 'x644',
       'x645', 'x646', 'x647', 'x648', 'x649', 'x650', 'x651', 'x652', 'x657',
       'x754', 'x893', 'x898', 'x902', 'x906', 'x907', 'x908', 'x909', 'x934',
       'x935', 'x939', 'x940', 'x941', 'x942', 'x1032', 'x1035', 'x1036',
       'x1143', 'x1147', 'x1150', 'x1159', 'x1160', 'x1161', 'x1162', 'x1181',
       'x1184', 'x1185']].copy()

## Uses vif removed columns data and missing value threshold data
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.33, random_state=42)

## Uses only missing value threshold data
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.33, random_state=42)

import nn
from metrics import generate_baselines

random_baseline, modal_baseline = generate_baselines(y_train, y_test)

model, losses, accs, mc_std = nn.train_PT(np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test))
steps = [i for i in range(len(losses))]

fig, axs = plt.subplots(3)
axs[0].plot(steps, losses)

axs[1].plot(steps, accs)
axs[1].axhline(y=random_baseline,c='r')
axs[1].axhline(y=modal_baseline,c='g')

axs[2].plot(steps, mc_std)

plt.show()
plt.close()