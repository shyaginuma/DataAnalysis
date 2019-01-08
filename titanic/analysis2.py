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
train_data_sub2.info()

train_data_sub2.describe()

plt.figure(figsize=(15,6))
plt.hist(train_data_sub2.Age[train_data_sub2.Survived == 0], normed=True, bins=10, alpha=0.8, color='red', label='not survived')
plt.hist(train_data_sub2.Age[train_data_sub2.Survived == 1], normed=True, bins=10, alpha=0.7, color='blue', label='survived')
plt.grid(True)
plt.legend(loc='best', fontsize=15)
plt.xlabel('Age')


