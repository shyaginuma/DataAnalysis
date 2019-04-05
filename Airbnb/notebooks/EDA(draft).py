# -*- coding: utf-8 -*-
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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan
from pandas_summary import DataFrameSummary

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# -

# ## Idea
#
# * 時系列のプロットで傾向を探る
# * レビューコメントのポジネガ分類
# * レビューコメントの多い時期

boston_calendar = pd.read_csv('Airbnb/boston-airbnb-open-data/calendar.csv')
boston_listing = pd.read_csv('Airbnb/boston-airbnb-open-data/listings.csv')
boston_review = pd.read_csv('Airbnb/boston-airbnb-open-data/reviews.csv')
seattle_calendar = pd.read_csv('Airbnb/seattle/calendar.csv')
seattle_listing = pd.read_csv('Airbnb/seattle/listings.csv')
seattle_review = pd.read_csv('Airbnb/seattle/reviews.csv')

print("listing_id ratio of boston_calendar: ", round(boston_calendar.listing_id.nunique()/len(boston_calendar), 3),
         "\nlisting_id counts: ", boston_calendar.listing_id.nunique())

print("listing_id ratio of boston_listing: ", boston_listing.id.nunique()/len(boston_listing))
print("listing_id counts: ", boston_listing.id.nunique())

print("listing_id ratio of boston_review: ", boston_review.listing_id.nunique()/len(boston_review))
print("listing_id counts: ", boston_review.listing_id.nunique())

boston_calendar[boston_calendar.listing_id == 3075044].shape

# listing_idごとに365日のデータが入っているっぽい

boston_groupby = boston_calendar.groupby('listing_id')["listing_id"].count()

boston_groupby[boston_groupby != 365]

# 一個だけ二年分データがあるやつがある

# ### calendar
# * listing_id毎に一年分のデータが入っている（一部二年分もあり）
# * availableがtだとpriceがある。fだとpriceはnull

boston_listing.head()

boston_listing.columns

# カラム大量

boston_review.head()

boston_review.comments.values[0]

seattle_review.head()

boston_calendar.head()

# +
# 家賃の時系列プロット

listing_sample = boston_calendar[(boston_calendar.listing_id == 3075044) & (boston_calendar.available == 't')]

plt.figure(figsize=(15, 7))
plt.plot(listing_sample.date.values[:14], listing_sample.price.values[:14])
# -

# 特定の曜日に10$上がってる？

listing_sample['price'] = listing_sample['price'].map(lambda x: float(x[1:]))
listing_sample['date'] = pd.to_datetime(listing_sample['date'])
listing_sample['weekday'] = listing_sample.date.dt.weekday
listing_sample['weekday_name'] = listing_sample.date.dt.weekday_name
listing_sample.head()

listing_sample.groupby('weekday_name')['price'].mean()

# 週末（金曜、土曜）が上がっている

# +
# カレンダーで貸し出し期間が少ない物件を探す

saled_days_count = boston_calendar.groupby('listing_id')['price'].count().reset_index()

plt.figure(figsize=(15, 8))
plt.hist(saled_days_count.price, 100)
plt.show()
# -

# 売り出し日が少ない物件が多い

# +
# 貸し出されている物件数 時系列プロット

saled_room_by_date = boston_calendar.groupby('date')['price'].count()

plt.figure(figsize=(15, 7))
plt.plot(saled_room_by_date.index[:300], saled_room_by_date.values[:300])
# -

def transform_calendar(calendar_df):

    #calendar_df['price'] = calendar_df['price'].map(lambda x: float(x[1:]))

    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    calendar_df['weekday'] = calendar_df.date.dt.weekday
    calendar_df['weekday_name'] = calendar_df.date.dt.weekday_name
    
    return calendar_df

# +
boston_calendar = transform_calendar(boston_calendar)
seattle_calendar = transform_calendar(seattle_calendar)

saled_room_by_date_boston = boston_calendar.groupby('date')['price'].count()
saled_room_by_date_seattle = seattle_calendar.groupby('date')['price'].count()

plt.figure(figsize=(15, 7))
plt.plot(saled_room_by_date_boston.index, saled_room_by_date_boston.values, label='boston')
plt.plot(saled_room_by_date_seattle.index, saled_room_by_date_seattle.values, label='seattle')
plt.legend()

# +
# レビュー数のトレンド

reviewed_by_date = boston_review.groupby('date')['reviewer_id'].count()

plt.figure(figsize=(15, 7))
plt.plot(reviewed_by_date.index, reviewed_by_date.values)

# +
reviewed_by_date = seattle_review.groupby('date')['reviewer_id'].count()

plt.figure(figsize=(15, 7))
plt.plot(reviewed_by_date.index, reviewed_by_date.values)
# -


