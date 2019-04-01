# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# %matplotlib inline

train_data = pd.read_csv('~/Downloads/all/train.csv')
test_data = pd.read_csv('~/Downloads/all/test.csv')
train_data.head()

train_data_sub1 = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
train_data_sub2 = train_data_sub1.dropna()
train_data_sub3 = pd.get_dummies(train_data_sub2)
train_data_sub3.head()

# +
X_train = train_data_sub3.drop('Survived', axis=1)
y_train = train_data_sub3.Survived

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# +
bench_mark = train_data_sub3.Survived.sum() / train_data_sub3.Survived.count()

print("bench_mark : ", bench_mark)
print("training score : ", log_reg.score(X_train, y_train))
# -


