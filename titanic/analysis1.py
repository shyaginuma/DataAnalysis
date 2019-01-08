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


